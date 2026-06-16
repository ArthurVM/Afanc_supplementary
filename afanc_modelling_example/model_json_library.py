import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "model_schema_v1"
HIERARCHY_ROOT_NODE = "__root__"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stringify_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _sorted_unique_strings(values: Iterable[Any]) -> List[str]:
    return sorted({str(value) for value in values})


def _children_by_parent(parent_map: Mapping[str, Optional[str]]) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = {}
    for child, parent in parent_map.items():
        if parent is None:
            continue
        children.setdefault(str(parent), []).append(str(child))
    for parent in children:
        children[parent] = sorted(children[parent])
    return dict(sorted(children.items()))


def _descendants_for_node(
    node: str,
    children_by_parent: Mapping[str, Sequence[str]],
) -> List[str]:
    ## iterative walk keeps hierarchy handling boring
    descendants: List[str] = []
    stack = [str(node)]
    while stack:
        current = stack.pop()
        descendants.append(current)
        stack.extend(reversed(list(children_by_parent.get(current, ()))))
    return descendants


def _ancestor_path_for_node(
    node: str,
    parent_map: Mapping[str, Optional[str]],
) -> List[str]:
    ## path excludes the synthetic root because it is not a lineage
    path: List[str] = []
    current: Optional[str] = str(node)
    seen: set[str] = set()
    while current is not None and current not in seen:
        seen.add(current)
        if current != HIERARCHY_ROOT_NODE:
            path.append(current)
        current = parent_map.get(current)
    return list(reversed(path))


def _top_classifier_node(root_lineages: Sequence[str]) -> str:
    roots = [str(root) for root in root_lineages]
    if len(roots) == 1:
        return roots[0]
    return HIERARCHY_ROOT_NODE


def _normalise_parent_map(parent_map: Mapping[Any, Any]) -> Dict[str, Optional[str]]:
    normalised: Dict[str, Optional[str]] = {}
    for child, parent in parent_map.items():
        child_id = str(child)
        parent_id = _stringify_or_none(parent)
        normalised[child_id] = parent_id
        ## mention parents even if they had no explicit row
        if parent_id is not None and parent_id not in normalised:
            normalised[parent_id] = None
    return dict(sorted(normalised.items()))


def _build_hierarchy(parent_map: Optional[Mapping[Any, Any]], topology: str) -> Dict[str, Any]:
    if topology == "flat":
        ## flat models carry the same keys with empty hierarchy values
        return {
            "parent_map": None,
            "children_by_parent": {},
            "root_lineages": [],
            "leaf_lineages": [],
            "top_classifier_node": None,
        }

    if not parent_map:
        raise ValueError("Hierarchical models require a non-empty parent_map.")

    normalised_parent_map = _normalise_parent_map(parent_map)
    root_lineages = sorted(
        [lineage for lineage, parent in normalised_parent_map.items() if parent is None]
    )
    children = _children_by_parent(normalised_parent_map)
    leaf_lineages = sorted(
        [lineage for lineage in normalised_parent_map if not children.get(lineage)]
    )
    top_node = _top_classifier_node(root_lineages)

    if top_node == HIERARCHY_ROOT_NODE:
        ## multiple roots get a synthetic parent for first pass inference
        parent_with_root = dict(normalised_parent_map)
        parent_with_root[HIERARCHY_ROOT_NODE] = None
        for root in root_lineages:
            parent_with_root[root] = HIERARCHY_ROOT_NODE
        normalised_parent_map = parent_with_root
        children = _children_by_parent(normalised_parent_map)

    return {
        "parent_map": normalised_parent_map,
        "children_by_parent": children,
        "root_lineages": root_lineages,
        "leaf_lineages": leaf_lineages,
        "top_classifier_node": top_node,
    }


def _base_model(
    *,
    model_id: str,
    species_id: str,
    model_type: str,
    feature_family: str,
    topology: str,
    emission_model: str,
    reference: Optional[Mapping[str, Any]] = None,
    generation: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    summary: Optional[Mapping[str, Any]] = None,
    parent_map: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    hierarchy = _build_hierarchy(parent_map=parent_map, topology=topology)
    ## every builder starts from this same schema shell
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": str(model_id),
        "model_type": str(model_type),
        "species_id": str(species_id),
        "architecture": {
            "feature_family": str(feature_family),
            "topology": str(topology),
            "emission_model": str(emission_model),
        },
        "reference": deepcopy(dict(reference or {})),
        "hierarchy": hierarchy,
        "lineages": [],
        "loci": [],
        "summary": deepcopy(dict(summary or {})),
        "generation": {
            "created_at": _utc_now_iso(),
            **deepcopy(dict(generation or {})),
        },
        "provenance": deepcopy(dict(provenance or {})),
    }


def _parse_allele_string(allele_id: str) -> Dict[str, Any]:
    parts = str(allele_id).rsplit(".", 3)
    if len(parts) != 4:
        raise ValueError(
            f"Allele ID {allele_id!r} must have format 'chrom.pos.ref.alt' if passed as a string."
        )
    chrom, pos, ref, alt = parts
    pos_int = int(pos)
    return {
        "allele_id": str(allele_id),
        "chrom": str(chrom),
        "pos": pos_int,
        "bed_start": pos_int - 1,
        "bed_end": pos_int,
        "ref": str(ref),
        "alt": str(alt),
    }


def _normalise_locus_spec(locus_spec: Any) -> Dict[str, Any]:
    if isinstance(locus_spec, str):
        ## string locus specs are dotted allele ids
        parsed = _parse_allele_string(locus_spec)
        parsed["locus_id"] = parsed["allele_id"]
        return parsed

    if not isinstance(locus_spec, Mapping):
        raise ValueError("Each locus spec must be a string allele ID or a mapping.")

    if "allele_id" in locus_spec:
        ## allele id wins and coordinate overrides can still follow
        parsed = _parse_allele_string(str(locus_spec["allele_id"]))
    else:
        required = {"chrom", "pos", "ref", "alt"}
        missing = required - set(locus_spec)
        if missing:
            raise ValueError(f"Locus spec is missing required keys: {sorted(missing)}")
        pos_int = int(locus_spec["pos"])
        parsed = {
            "allele_id": f"{locus_spec['chrom']}.{pos_int}.{locus_spec['ref']}.{locus_spec['alt']}",
            "chrom": str(locus_spec["chrom"]),
            "pos": pos_int,
            "bed_start": pos_int - 1,
            "bed_end": pos_int,
            "ref": str(locus_spec["ref"]),
            "alt": str(locus_spec["alt"]),
        }

    locus_id = str(locus_spec.get("locus_id", parsed["allele_id"]))
    parsed["locus_id"] = locus_id
    parsed["bed_start"] = int(locus_spec.get("bed_start", parsed["bed_start"]))
    parsed["bed_end"] = int(locus_spec.get("bed_end", parsed["bed_end"]))
    return parsed


def _normalise_empirical_locus(locus_spec: Mapping[str, Any]) -> Dict[str, Any]:
    locus = _normalise_locus_spec(locus_spec)
    target_lineage = locus_spec.get("target_lineage")
    if target_lineage is None:
        raise ValueError("Empirical loci require 'target_lineage'.")

    emission = locus_spec.get("emission")
    if emission is None:
        ## old models used empirical_bayes for the same block
        emission = locus_spec.get("empirical_bayes")
    if emission is None:
        raise ValueError("Empirical loci require an 'emission' block.")

    locus["target_lineage"] = str(target_lineage)
    locus["emission"] = {
        "target_frequency": float(emission["target_frequency"]),
        "background_frequency": float(emission["background_frequency"]),
    }
    return locus


def _canonical_locus_record(
    locus_spec: Mapping[str, Any],
    lineage_id: str,
    *,
    role: str = "required",
) -> Dict[str, Any]:
    locus = _normalise_locus_spec(locus_spec)
    locus["target_lineage"] = str(lineage_id)
    locus["canonical"] = {
        "role": str(role),
    }
    return locus


def _build_lineage_records(
    lineages: Sequence[str],
    *,
    priors: Optional[Mapping[Any, Any]] = None,
    parent_map: Optional[Mapping[str, Optional[str]]] = None,
    direct_locus_ids_by_lineage: Optional[Mapping[str, Sequence[str]]] = None,
    inherited_locus_ids_by_lineage: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[Dict[str, Any]]:
    priors = priors or {}
    direct_locus_ids_by_lineage = direct_locus_ids_by_lineage or {}
    inherited_locus_ids_by_lineage = inherited_locus_ids_by_lineage or {}

    ## lineage records keep both direct and inherited marker views
    lineage_records: List[Dict[str, Any]] = []
    for lineage_id in sorted({str(lineage) for lineage in lineages}):
        direct_ids = _sorted_unique_strings(direct_locus_ids_by_lineage.get(lineage_id, ()))
        inherited_ids = _sorted_unique_strings(
            inherited_locus_ids_by_lineage.get(lineage_id, direct_ids)
        )
        record: Dict[str, Any] = {
            "lineage_id": lineage_id,
            "prior": float(priors.get(lineage_id, 0.0)),
            "locus_ids": inherited_ids,
            "direct_locus_ids": direct_ids,
            "marker_count": len(inherited_ids),
        }
        if parent_map is not None:
            record["parent_lineage"] = _stringify_or_none(parent_map.get(lineage_id))
        lineage_records.append(record)
    return lineage_records


def build_empirical_loso_model(
    *,
    model_id: str,
    species_id: str,
    lineages: Sequence[Mapping[str, Any]],
    loci: Sequence[Mapping[str, Any]],
    emission_model: str,
    reference: Optional[Mapping[str, Any]] = None,
    parent_map: Optional[Mapping[Any, Any]] = None,
    generation_metadata: Optional[Mapping[str, Any]] = None,
    builder_parameters: Optional[Mapping[str, Any]] = None,
    summary: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if emission_model not in {"empirical_bayes", "semi_bayes", "full_bayes"}:
        raise ValueError("Empirical models require emission_model in {'empirical_bayes', 'semi_bayes', 'full_bayes'}.")

    ## LOSO output can be flat or hierarchical but keeps one schema
    topology = "hierarchical" if parent_map else "flat"
    model = _base_model(
        model_id=model_id,
        species_id=species_id,
        model_type="empirical_loso",
        feature_family="empirical",
        topology=topology,
        emission_model=emission_model,
        reference=reference,
        generation={
            "builder": "loso",
            "source_type": "loso_results",
            "parameters": deepcopy(dict(builder_parameters or {})),
            "metadata": deepcopy(dict(generation_metadata or {})),
        },
        provenance=provenance,
        summary=summary,
        parent_map=parent_map,
    )

    normalised_loci = [_normalise_empirical_locus(locus) for locus in loci]
    model["loci"] = normalised_loci

    lineage_ids = _sorted_unique_strings(lineage["lineage_id"] for lineage in lineages)
    normalised_parent_map = model["hierarchy"]["parent_map"]
    if topology == "hierarchical" and normalised_parent_map is not None:
        lineage_ids = _sorted_unique_strings(
            set(lineage_ids) | {node for node in normalised_parent_map if node != HIERARCHY_ROOT_NODE}
        )

    lineage_records_input = {str(lineage["lineage_id"]): dict(lineage) for lineage in lineages}
    direct_ids_by_lineage: Dict[str, List[str]] = {}
    for locus in normalised_loci:
        direct_ids_by_lineage.setdefault(str(locus["target_lineage"]), []).append(str(locus["locus_id"]))

    inherited_ids_by_lineage: Dict[str, List[str]] = {}
    if topology == "hierarchical":
        ## descendants inherit marker ids upward for lineage records
        children = model["hierarchy"]["children_by_parent"]
        for lineage_id in lineage_ids:
            if lineage_id == HIERARCHY_ROOT_NODE:
                continue
            descendant_nodes = _descendants_for_node(lineage_id, children)
            inherited: List[str] = []
            for descendant in descendant_nodes:
                inherited.extend(direct_ids_by_lineage.get(descendant, ()))
            inherited_ids_by_lineage[lineage_id] = _sorted_unique_strings(inherited)
    else:
        inherited_ids_by_lineage = {
            lineage_id: _sorted_unique_strings(direct_ids_by_lineage.get(lineage_id, ()))
            for lineage_id in lineage_ids
        }

    priors = {
        lineage_id: float(lineage_records_input.get(lineage_id, {}).get("prior", 0.0))
        for lineage_id in lineage_ids
        if lineage_id != HIERARCHY_ROOT_NODE
    }

    model["lineages"] = _build_lineage_records(
        [lineage_id for lineage_id in lineage_ids if lineage_id != HIERARCHY_ROOT_NODE],
        priors=priors,
        parent_map=None if topology == "flat" else model["hierarchy"]["parent_map"],
        direct_locus_ids_by_lineage=direct_ids_by_lineage,
        inherited_locus_ids_by_lineage=inherited_ids_by_lineage,
    )

    model["summary"].setdefault("lineage_count", len(model["lineages"]))
    model["summary"].setdefault("locus_count", len(model["loci"]))
    return model


def _canonical_loci_from_map(
    canonical_snp_map: Mapping[Any, Sequence[Any]],
    *,
    role: str = "required",
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    loci: List[Dict[str, Any]] = []
    direct_ids_by_lineage: Dict[str, List[str]] = {}
    seen: set[Tuple[str, str]] = set()

    ## de duplicate only within a lineage target
    for lineage_id, locus_specs in canonical_snp_map.items():
        lineage_name = str(lineage_id)
        direct_ids_by_lineage.setdefault(lineage_name, [])
        for locus_spec in locus_specs:
            locus = _canonical_locus_record(locus_spec, lineage_name, role=role)
            dedupe_key = (lineage_name, str(locus["locus_id"]))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            loci.append(locus)
            direct_ids_by_lineage[lineage_name].append(str(locus["locus_id"]))

    loci.sort(key=lambda locus: (str(locus["target_lineage"]), str(locus["chrom"]), int(locus["pos"]), str(locus["alt"])))
    for lineage_id in direct_ids_by_lineage:
        direct_ids_by_lineage[lineage_id] = _sorted_unique_strings(direct_ids_by_lineage[lineage_id])
    return loci, direct_ids_by_lineage


def build_flat_canonical_model(
    *,
    model_id: str,
    species_id: str,
    canonical_snp_map: Mapping[Any, Sequence[Any]],
    priors: Optional[Mapping[Any, Any]] = None,
    reference: Optional[Mapping[str, Any]] = None,
    generation_metadata: Optional[Mapping[str, Any]] = None,
    builder_parameters: Optional[Mapping[str, Any]] = None,
    summary: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    loci, direct_ids_by_lineage = _canonical_loci_from_map(canonical_snp_map)
    lineage_ids = _sorted_unique_strings(canonical_snp_map.keys())
    if priors is None:
        ## flat canonical defaults to uniform priors across named lineages
        uniform = 1.0 / max(len(lineage_ids), 1)
        priors = {lineage_id: uniform for lineage_id in lineage_ids}

    model = _base_model(
        model_id=model_id,
        species_id=species_id,
        model_type="canonical",
        feature_family="canonical",
        topology="flat",
        emission_model="none",
        reference=reference,
        generation={
            "builder": "canonical",
            "source_type": "canonical_snp_map",
            "parameters": deepcopy(dict(builder_parameters or {})),
            "metadata": deepcopy(dict(generation_metadata or {})),
        },
        provenance=provenance,
        summary=summary,
    )
    model["loci"] = loci
    model["lineages"] = _build_lineage_records(
        lineage_ids,
        priors=priors,
        direct_locus_ids_by_lineage=direct_ids_by_lineage,
        inherited_locus_ids_by_lineage=direct_ids_by_lineage,
    )
    model["summary"].setdefault("lineage_count", len(model["lineages"]))
    model["summary"].setdefault("locus_count", len(model["loci"]))
    return model


def build_hierarchical_canonical_model(
    *,
    model_id: str,
    species_id: str,
    canonical_snp_map: Mapping[Any, Sequence[Any]],
    parent_map: Mapping[Any, Any],
    priors: Optional[Mapping[Any, Any]] = None,
    reference: Optional[Mapping[str, Any]] = None,
    generation_metadata: Optional[Mapping[str, Any]] = None,
    builder_parameters: Optional[Mapping[str, Any]] = None,
    summary: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    loci, direct_ids_by_lineage = _canonical_loci_from_map(canonical_snp_map)
    ## hierarchy is resolved before lineage records inherit markers
    model = _base_model(
        model_id=model_id,
        species_id=species_id,
        model_type="canonical",
        feature_family="canonical",
        topology="hierarchical",
        emission_model="none",
        reference=reference,
        generation={
            "builder": "canonical",
            "source_type": "canonical_snp_map",
            "parameters": deepcopy(dict(builder_parameters or {})),
            "metadata": deepcopy(dict(generation_metadata or {})),
        },
        provenance=provenance,
        summary=summary,
        parent_map=parent_map,
    )
    model["loci"] = loci

    hierarchy_parent_map = dict(model["hierarchy"]["parent_map"] or {})
    lineage_ids = _sorted_unique_strings(
        set(canonical_snp_map.keys())
        | {node for node in hierarchy_parent_map if node != HIERARCHY_ROOT_NODE}
    )
    if priors is None:
        leaf_lineages = [
            lineage_id
            for lineage_id in model["hierarchy"]["leaf_lineages"]
            if lineage_id in lineage_ids
        ]
        uniform = 1.0 / max(len(leaf_lineages), 1)
        priors = {lineage_id: uniform for lineage_id in leaf_lineages}

    inherited_ids_by_lineage: Dict[str, List[str]] = {}
    for lineage_id in lineage_ids:
        ## canonical markers inherit from ancestors down to leaves
        ancestor_nodes = _ancestor_path_for_node(lineage_id, hierarchy_parent_map)
        inherited: List[str] = []
        for ancestor in ancestor_nodes:
            inherited.extend(direct_ids_by_lineage.get(ancestor, ()))
        inherited_ids_by_lineage[lineage_id] = _sorted_unique_strings(inherited)

    model["lineages"] = _build_lineage_records(
        lineage_ids,
        priors=priors,
        parent_map=hierarchy_parent_map,
        direct_locus_ids_by_lineage=direct_ids_by_lineage,
        inherited_locus_ids_by_lineage=inherited_ids_by_lineage,
    )
    model["summary"].setdefault("lineage_count", len(model["lineages"]))
    model["summary"].setdefault("locus_count", len(model["loci"]))
    return model


def build_canonical_as_empirical_model(
    *,
    model_id: str,
    species_id: str,
    canonical_snp_map: Mapping[Any, Sequence[Any]],
    parent_map: Optional[Mapping[Any, Any]] = None,
    epsilon: float = 1e-6,
    priors: Optional[Mapping[Any, Any]] = None,
    reference: Optional[Mapping[str, Any]] = None,
    generation_metadata: Optional[Mapping[str, Any]] = None,
    builder_parameters: Optional[Mapping[str, Any]] = None,
    summary: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    epsilon = float(epsilon)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be > 0 and < 0.5.")

    ## this turns hard canonical calls into near deterministic emissions
    topology = "hierarchical" if parent_map else "flat"
    model = _base_model(
        model_id=model_id,
        species_id=species_id,
        model_type="canonical_as_empirical",
        feature_family="empirical",
        topology=topology,
        emission_model="semi_bayes",
        reference=reference,
        generation={
            "builder": "canonical_as_empirical",
            "source_type": "canonical_snp_map",
            "parameters": {
                "epsilon": epsilon,
                **deepcopy(dict(builder_parameters or {})),
            },
            "metadata": deepcopy(dict(generation_metadata or {})),
        },
        provenance=provenance,
        summary=summary,
        parent_map=parent_map,
    )

    canonical_loci, direct_ids_by_lineage = _canonical_loci_from_map(canonical_snp_map)
    loci = []
    for locus in canonical_loci:
        ## keep canonical metadata but add empirical frequencies
        empirical_locus = deepcopy(locus)
        empirical_locus["emission"] = {
            "target_frequency": 1.0 - epsilon,
            "background_frequency": epsilon,
        }
        empirical_locus["canonical_as_empirical"] = {
            "epsilon": epsilon,
        }
        loci.append(empirical_locus)
    model["loci"] = loci

    if topology == "hierarchical":
        hierarchy_parent_map = dict(model["hierarchy"]["parent_map"] or {})
        lineage_ids = _sorted_unique_strings(
            set(canonical_snp_map.keys())
            | {node for node in hierarchy_parent_map if node != HIERARCHY_ROOT_NODE}
        )
    else:
        hierarchy_parent_map = None
        lineage_ids = _sorted_unique_strings(canonical_snp_map.keys())

    if priors is None:
        uniform = 1.0 / max(len(lineage_ids), 1)
        priors = {lineage_id: uniform for lineage_id in lineage_ids}

    inherited_ids_by_lineage: Dict[str, List[str]] = {}
    if topology == "hierarchical" and hierarchy_parent_map is not None:
        for lineage_id in lineage_ids:
            ancestor_nodes = _ancestor_path_for_node(lineage_id, hierarchy_parent_map)
            inherited: List[str] = []
            for ancestor in ancestor_nodes:
                inherited.extend(direct_ids_by_lineage.get(ancestor, ()))
            inherited_ids_by_lineage[lineage_id] = _sorted_unique_strings(inherited)
    else:
        inherited_ids_by_lineage = {
            lineage_id: _sorted_unique_strings(direct_ids_by_lineage.get(lineage_id, ()))
            for lineage_id in lineage_ids
        }

    model["lineages"] = _build_lineage_records(
        lineage_ids,
        priors=priors,
        parent_map=hierarchy_parent_map,
        direct_locus_ids_by_lineage=direct_ids_by_lineage,
        inherited_locus_ids_by_lineage=inherited_ids_by_lineage,
    )
    model["summary"].setdefault("lineage_count", len(model["lineages"]))
    model["summary"].setdefault("locus_count", len(model["loci"]))
    model["summary"].setdefault("epsilon", epsilon)
    return model


def validate_model_schema(model: Mapping[str, Any]) -> None:
    required_top_level = {
        "schema_version",
        "model_id",
        "model_type",
        "species_id",
        "architecture",
        "reference",
        "hierarchy",
        "lineages",
        "loci",
        "summary",
        "generation",
        "provenance",
    }
    missing = required_top_level - set(model)
    if missing:
        raise ValueError(f"Model is missing required top-level keys: {sorted(missing)}")

    if model["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {model['schema_version']!r}")

    architecture = model["architecture"]
    for key in ("feature_family", "topology", "emission_model"):
        if key not in architecture:
            raise ValueError(f"Model architecture is missing {key!r}.")

    if architecture["feature_family"] not in {"empirical", "canonical"}:
        raise ValueError("feature_family must be 'empirical' or 'canonical'.")
    if architecture["topology"] not in {"flat", "hierarchical"}:
        raise ValueError("topology must be 'flat' or 'hierarchical'.")

    if architecture["feature_family"] == "empirical":
        if architecture["emission_model"] not in {"empirical_bayes", "semi_bayes", "full_bayes"}:
            raise ValueError("Empirical models have invalid emission_model.")
    else:
        if architecture["emission_model"] != "none":
            raise ValueError("Canonical models must use emission_model='none'.")

    for lineage in model["lineages"]:
        for key in ("lineage_id", "prior", "locus_ids", "marker_count"):
            if key not in lineage:
                raise ValueError(f"Lineage record is missing {key!r}.")

    for locus in model["loci"]:
        for key in ("locus_id", "allele_id", "chrom", "pos", "ref", "alt", "target_lineage"):
            if key not in locus:
                raise ValueError(f"Locus record is missing {key!r}.")
        if architecture["feature_family"] == "empirical":
            emission = locus.get("emission")
            if emission is None:
                raise ValueError("Empirical loci require an 'emission' block.")
            for key in ("target_frequency", "background_frequency"):
                if key not in emission:
                    raise ValueError(f"Empirical locus emission block is missing {key!r}.")
        else:
            canonical = locus.get("canonical")
            if canonical is None:
                raise ValueError("Canonical loci require a 'canonical' block.")
            if "role" not in canonical:
                raise ValueError("Canonical locus block is missing 'role'.")


def write_model_json(model: Mapping[str, Any], output_json: str) -> None:
    validate_model_schema(model)
    path = Path(output_json)
    ## write only after schema validation has passed
    with path.open("w") as handle:
        json.dump(model, handle, indent=2)


def load_model_json(model_json: str) -> Dict[str, Any]:
    path = Path(model_json)
    with path.open() as handle:
        model = json.load(handle)
    validate_model_schema(model)
    return model
