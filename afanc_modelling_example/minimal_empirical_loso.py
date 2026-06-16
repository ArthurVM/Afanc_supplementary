import json
import math
import re
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from model_json_library import build_empirical_loso_model, write_model_json as write_schema_model_json


LosoFold = namedtuple("LosoFold", ["group", "test_ids", "train_ids"])
DEFAULT_COOC_THRESHOLD = 0.98


def _safe_prob(value, eps=1e-12):
    return min(max(float(value), eps), 1.0 - eps)


def _metric_score(y_true, y_pred, prob_df, metric):
    if metric == "macro_f1":
        return f1_score(y_true, y_pred, average="macro")
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "log_loss":
        labels = sorted(pd.unique(y_true))
        return -log_loss(y_true, prob_df[labels], labels=labels)
    raise ValueError(f"Unsupported metric {metric!r}.")


def _make_loso_folds(meta, sample_col, group_col):
    folds = []
    for group_value, group_df in meta.groupby(group_col, sort=True):
        test_ids = [str(sample_id) for sample_id in group_df[sample_col]]
        train_ids = [str(sample_id) for sample_id in meta.loc[meta[group_col] != group_value, sample_col]]
        
        if not test_ids or not train_ids:
            continue
        
        train_lineages = set(meta.loc[meta[sample_col].isin(train_ids), "___lineage"])
        test_lineages = set(meta.loc[meta[sample_col].isin(test_ids), "___lineage"])
        
        ## LOSO only counts when training can name every test lineage
        if test_lineages.issubset(train_lineages):
            folds.append(LosoFold(str(group_value), test_ids, train_ids))
            
    return folds


def _subset_ardal(ard_obj, sample_ids, drop_zero_cols=True):
    return ard_obj.get.subset(
        guids=list(sample_ids),
        chunk_size=1000,
        threads=12,
        drop_zero_cols=drop_zero_cols,
        child_ardal_kwargs={"roaring": True},
    )


def _get_sample_allele_sets(ard_obj, sample_ids, alleles, backend="auto"):
    guid_to_alleles = ard_obj.get.subset(
        guids=list(sample_ids),
        alleles=list(alleles),
        chunk_size=1000,
        threads=12,
        drop_zero_cols=True,
        child_ardal_kwargs={"roaring": True},
    ).io.to_dict(backend=backend)

    sample_alleles = {}
    for sample_id in sample_ids:
        sample_alleles[str(sample_id)] = set(guid_to_alleles.get(sample_id, []))
    return sample_alleles


def _rank_candidate_alleles(
    ard_train,
    train_meta,
    sample_col,
    top_k_per_lineage,
    include_negative_alleles=False,
    top_k_negative_per_lineage=None,
):
    ranked_by_lineage = {}
    for lineage in sorted(train_meta["___lineage"].unique()):
        ## start with markers enriched in the target lineage
        positives = train_meta.loc[train_meta["___lineage"] == lineage, sample_col].tolist()
        ranked_items = list(
            ard_train.stats.allele_inform(positives, method="kullbackleibler").items()
        )[: int(top_k_per_lineage)]
        lineage_ranked = [
            {
                "allele_id": str(allele_id),
                "score": float(score),
                "target_lineage": str(lineage),
                "evidence_direction": "positive",
            }
            for allele_id, score in ranked_items
        ]

        if include_negative_alleles:
            ## optional negative markers come from the not this lineage side
            negatives = train_meta.loc[train_meta["___lineage"] != lineage, sample_col].tolist()
            negative_limit = top_k_negative_per_lineage
            if negative_limit is None:
                negative_limit = top_k_per_lineage
            negative_ranked_items = list(
                ard_train.stats.allele_inform(negatives, method="kullbackleibler").items()
            )[: int(negative_limit)]
            seen = {record["allele_id"] for record in lineage_ranked}
            for allele_id, score in negative_ranked_items:
                allele_id = str(allele_id)
                if allele_id in seen:
                    continue
                lineage_ranked.append(
                    {
                        "allele_id": allele_id,
                        "score": float(score),
                        "target_lineage": str(lineage),
                        "evidence_direction": "negative",
                    }
                )
                seen.add(allele_id)

        ranked_by_lineage[str(lineage)] = lineage_ranked
    return ranked_by_lineage


def _round_robin_candidates(ranked_by_lineage, max_model_alleles):
    merged = []
    seen = set()
    lineages = sorted(ranked_by_lineage)
    depth = 0

    ## walk ranks across lineages so big classes do not dominate
    while len(merged) < int(max_model_alleles):
        added_any = False
        for lineage in lineages:
            ranked = ranked_by_lineage[lineage]
            if depth >= len(ranked):
                continue
            record = ranked[depth]
            allele_id = str(record["allele_id"])
            if allele_id in seen:
                continue
            seen.add(allele_id)
            merged.append(dict(record))
            added_any = True
            if len(merged) >= int(max_model_alleles):
                break
        if not added_any:
            break
        depth += 1

    return merged


def _prune_ranked_by_lineage_cooc(ard_subset, ranked_by_lineage, cooc_threshold, threads=12):
    if cooc_threshold is None:
        return ranked_by_lineage

    pruned = {}
    for lineage, ranked_items in ranked_by_lineage.items():
        lineage_alleles = [str(record["allele_id"]) for record in ranked_items]
        if len(lineage_alleles) < 2:
            pruned[lineage] = [dict(record) for record in ranked_items]
            continue

        ## within each lineage keep the first marker from tight cooc blocks
        cooc_map = ard_subset.stats.allele_cooc(
            lineage_alleles,
            threshold=float(cooc_threshold),
            threads=int(threads),
        )
        cooc_map = {str(allele): {str(x) for x in partners} for allele, partners in cooc_map.items()}

        kept = []
        kept_ids = set()
        for record in ranked_items:
            allele_id = str(record["allele_id"])
            redundant = False
            for kept_id in kept_ids:
                if allele_id in cooc_map.get(kept_id, set()) or kept_id in cooc_map.get(allele_id, set()):
                    redundant = True
                    break
            if redundant:
                continue
            kept.append(dict(record))
            kept_ids.add(allele_id)

        pruned[str(lineage)] = kept

    return pruned


def _lineage_marker_counts(selected_alleles, allele_target_support, lineages):
    counts = Counter()
    for allele_id in selected_alleles:
        lineage_counts = allele_target_support.get(allele_id)
        if not lineage_counts:
            continue
        counts[lineage_counts.most_common(1)[0][0]] += 1
    return {str(lineage): int(counts.get(lineage, 0)) for lineage in sorted(lineages)}


def _consensus_target_lineages(allele_target_support):
    consensus = {}
    for allele_id, counts in allele_target_support.items():
        if not counts:
            continue
        consensus[str(allele_id)] = counts.most_common(1)[0][0]
    return consensus


def _stable_allele_order(alleles, allele_support, allele_target_support):
    consensus = _consensus_target_lineages(allele_target_support)
    return sorted(
        {str(allele_id) for allele_id in alleles},
        key=lambda allele_id: (
            int(allele_support.get(allele_id, 0)),
            int(max(allele_target_support.get(allele_id, Counter()).values(), default=0)),
            str(consensus.get(allele_id, "")),
            str(allele_id),
        ),
        reverse=True,
    )


def _limit_maximum_lineage_alleles(
    selected_alleles,
    allele_target_support,
    max_alleles_per_lineage,
):
    if int(max_alleles_per_lineage) <= 0:
        return list(selected_alleles)

    ## cap by consensus lineage after support ordering
    consensus = _consensus_target_lineages(allele_target_support)
    kept = []
    lineage_counts = Counter()
    for allele_id in selected_alleles:
        lineage = consensus.get(allele_id)
        if lineage is None:
            kept.append(allele_id)
            continue
        if lineage_counts[lineage] >= int(max_alleles_per_lineage):
            continue
        kept.append(allele_id)
        lineage_counts[lineage] += 1
    return kept


def _supplement_minimum_lineage_alleles_from_loso(
    selected_alleles,
    lineages,
    allele_support,
    allele_target_support,
    min_alleles_per_lineage,
):
    if int(min_alleles_per_lineage) <= 0:
        return list(selected_alleles)

    ## top up only from alleles that earned LOSO support somewhere
    consensus = _consensus_target_lineages(allele_target_support)
    supplemented = list(selected_alleles)
    selected = set(supplemented)
    counts = Counter(_lineage_marker_counts(supplemented, allele_target_support, lineages))

    for lineage in sorted(lineages):
        needed = int(min_alleles_per_lineage) - int(counts.get(lineage, 0))
        if needed <= 0:
            continue

        lineage_candidates = [
            allele_id
            for allele_id, target_lineage in consensus.items()
            if target_lineage == str(lineage) and allele_id not in selected
        ]
        lineage_candidates.sort(
            key=lambda allele_id: (
                int(allele_target_support.get(allele_id, Counter()).get(str(lineage), 0)),
                int(allele_support.get(allele_id, 0)),
                str(allele_id),
            ),
            reverse=True,
        )

        for allele_id in lineage_candidates:
            if allele_id in selected:
                continue
            supplemented.append(allele_id)
            selected.add(allele_id)
            counts[lineage] += 1
            if counts[lineage] >= int(min_alleles_per_lineage):
                break

    return supplemented


def _minimum_lineage_alleles_deficits(final_model, min_alleles_per_lineage):
    if int(min_alleles_per_lineage) <= 0:
        return {}

    lineage_marker_counts = {
        record["lineage_id"]: int(record["marker_count"])
        for record in final_model["lineages"]
    }
    return {
        lineage: count
        for lineage, count in lineage_marker_counts.items()
        if count < int(min_alleles_per_lineage)
    }


def _parse_ardal_allele(allele_id):
    allele_id = str(allele_id)
    
    ## accept the common allele id shapes we get from Ardal exports
    patterns = [
        r"^(?P<chrom>.+):(?P<pos>\d+):(?P<ref>[A-ZN]+)>(?P<alt>[A-ZN]+)$",
        r"^(?P<chrom>.+)_(?P<pos>\d+)_(?P<ref>[A-ZN]+)_(?P<alt>[A-ZN]+)$",
        r"^(?P<chrom>.+)\.(?P<pos>\d+)\.(?P<ref>[A-ZN]+)\.(?P<alt>[A-ZN]+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, allele_id)
        if match:
            chrom = str(match.group("chrom"))
            pos = int(match.group("pos"))
            ref = match.group("ref")
            alt = match.group("alt")
            return {
                "chrom": chrom,
                "pos": pos,
                "bed_start": pos - 1,
                "bed_end": pos,
                "ref": ref,
                "alt": alt,
            }

    return {
        "chrom": allele_id,
        "pos": 0,
        "bed_start": 0,
        "bed_end": 0,
        "ref": None,
        "alt": "N",
    }


def build_sparse_empirical_model_from_sets(
    sample_alleles,
    labels_by_sample,
    selected_alleles,
    species_id="unknown",
    model_id="minimal_empirical_loso_model",
    smoothing_alpha=1.0,
    uniform_priors=False,
    forced_target_lineages=None,
):
    lineage_to_samples = defaultdict(set)
    for sample_id, lineage in labels_by_sample.items():
        lineage_to_samples[str(lineage)].add(str(sample_id))

    lineages = sorted(lineage_to_samples)
    all_samples = set(labels_by_sample)
    lineage_counts = {lineage: len(sample_ids) for lineage, sample_ids in lineage_to_samples.items()}
    total_samples = sum(lineage_counts.values())

    if uniform_priors:
        priors = {lineage: 1.0 / len(lineages) for lineage in lineages}
    else:
        priors = {lineage: lineage_counts[lineage] / total_samples for lineage in lineages}

    loci = []
    locus_ids_by_lineage = defaultdict(list)

    for allele_id in selected_alleles:
        ## empirical model is just smoothed target versus pooled background
        carriers = {
            sample_id
            for sample_id, allele_set in sample_alleles.items()
            if allele_id in allele_set
        }
        lineage_freqs = {}
        for lineage in lineages:
            lineage_ids = lineage_to_samples[lineage]
            n = len(lineage_ids)
            k = len(carriers & lineage_ids)
            lineage_freqs[lineage] = float((k + smoothing_alpha) / (n + 2.0 * smoothing_alpha))

        target_lineage = None
        if forced_target_lineages is not None:
            ## keep fold selected lineage assignments stable in the final refit
            target_lineage = forced_target_lineages.get(str(allele_id))
            if target_lineage is not None:
                target_lineage = str(target_lineage)
                if target_lineage not in lineage_to_samples:
                    raise ValueError(
                        f"Forced target lineage {target_lineage!r} for allele {allele_id!r} is not present."
                    )
        if target_lineage is None:
            target_lineage = max(
                sorted(lineages),
                key=lambda lineage: (lineage_freqs[lineage], lineage),
            )
        background_ids = all_samples - lineage_to_samples[target_lineage]
        background_k = len(carriers & background_ids)
        background_n = len(background_ids)
        background_freq = float((background_k + smoothing_alpha) / (background_n + 2.0 * smoothing_alpha))

        allele_meta = _parse_ardal_allele(allele_id)
        locus_id = f"{allele_meta['chrom']}:{allele_meta['pos']}:{target_lineage}"
        loci.append(
            {
                "locus_id": locus_id,
                "allele_id": str(allele_id),
                "chrom": allele_meta["chrom"],
                "pos": allele_meta["pos"],
                "bed_start": allele_meta["bed_start"],
                "bed_end": allele_meta["bed_end"],
                "ref": allele_meta["ref"],
                "alt": allele_meta["alt"],
                "target_lineage": target_lineage,
                "emission": {
                    "target_frequency": float(lineage_freqs[target_lineage]),
                    "background_frequency": background_freq,
                },
            }
        )
        locus_ids_by_lineage[target_lineage].append(locus_id)

    lineage_records = []
    for lineage in lineages:
        locus_ids = sorted(locus_ids_by_lineage.get(lineage, []))
        lineage_records.append(
            {
                "lineage_id": lineage,
                "prior": float(priors[lineage]),
                "locus_ids": locus_ids,
                "marker_count": len(locus_ids),
            }
        )

    return build_empirical_loso_model(
        model_id=model_id,
        species_id=species_id,
        lineages=lineage_records,
        loci=loci,
        emission_model="empirical_bayes",
        summary={
            "lineage_count": len(lineage_records),
            "selected_locus_count_after_loso": len(loci),
        },
        generation_metadata={
            "builder_function": "build_sparse_empirical_model_from_sets",
        },
        builder_parameters={
            "smoothing_alpha": float(smoothing_alpha),
            "uniform_priors": bool(uniform_priors),
        },
        provenance={
            "created_by": "build_geolineage_min_model_sparse",
            "smoothing_alpha": float(smoothing_alpha),
            "uniform_priors": bool(uniform_priors),
        },
    )


def score_samples_sparse_tables(sample_alleles, model):
    priors = {record["lineage_id"]: float(record["prior"]) for record in model["lineages"]}
    lineages = [record["lineage_id"] for record in model["lineages"]]
    loci = model["loci"]

    records = {}
    for sample_id, allele_set in sample_alleles.items():
        log_scores = {}
        for lineage in lineages:
            logp = math.log(_safe_prob(priors[lineage]))
            
            ## score both seen and absent markers so silence still matters
            for locus in loci:
                emission = locus.get("emission", locus.get("empirical_bayes"))
                if emission is None:
                    raise ValueError(f"Locus {locus.get('locus_id', '?')!r} is missing an emission block.")
                p = emission["target_frequency"]
                if lineage != locus["target_lineage"]:
                    p = emission["background_frequency"]
                p = _safe_prob(p)
                allele_id = locus["allele_id"]
                logp += math.log(p) if allele_id in allele_set else math.log(1.0 - p)
            log_scores[lineage] = logp
        records[str(sample_id)] = log_scores

    score_df = pd.DataFrame.from_dict(records, orient="index")
    max_scores = score_df.max(axis=1)
    prob_df = np.exp(score_df.sub(max_scores, axis=0))
    prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)
    return score_df, prob_df


def build_lineage_min_model_sparse(
    ard_obj,
    meta_df,
    sample_col="Sample",
    lineage_col="Lineage",
    group_col="Country",
    top_k_per_lineage=200,
    max_model_alleles=500,
    min_lineage_samples=10,
    min_fold_support=0.5,
    smoothing_alpha=1.0,
    uniform_priors=False,
    metric="macro_f1",
    species_id="unknown",
    model_id="minimal_empirical_loso_model",
    score_tolerance=0.005,
    min_alleles_per_lineage=0,
    max_alleles_per_lineage=0,
    cooc_threshold=DEFAULT_COOC_THRESHOLD,
    include_negative_alleles=False,
    top_k_negative_per_lineage=None,
):
    if int(max_alleles_per_lineage) > 0 and int(min_alleles_per_lineage) > int(max_alleles_per_lineage):
        raise ValueError("max_alleles_per_lineage must be >= min_alleles_per_lineage.")

    ## normalise metadata once so folds and labels stay aligned
    meta = meta_df[[sample_col, lineage_col, group_col]].dropna().copy()
    meta[sample_col] = meta[sample_col].astype(str)
    meta["___lineage"] = meta[lineage_col].astype(str)
    meta[group_col] = meta[group_col].astype(str)

    lineage_counts = meta["___lineage"].value_counts()
    keep_lineages = set(lineage_counts[lineage_counts >= int(min_lineage_samples)].index)
    meta = meta[meta["___lineage"].isin(keep_lineages)].copy()
    if meta.empty:
        raise ValueError("No samples remain after applying min_lineage_samples.")

    folds = _make_loso_folds(meta, sample_col=sample_col, group_col=group_col)
    if not folds:
        raise ValueError("No valid LOSO folds were produced.")

    labels_by_sample = dict(zip(meta[sample_col], meta["___lineage"]))
    allele_support = Counter()
    allele_target_support = defaultdict(Counter)
    allele_direction_support = defaultdict(Counter)
    fold_results = []

    for fold in folds:
        print(f"Fold={fold.group}; n_test_ids={len(fold.test_ids)}; n_train_ids={len(fold.train_ids)}")
        train_meta = meta[meta[sample_col].isin(fold.train_ids)].copy()
        ard_train = _subset_ardal(ard_obj, fold.train_ids, drop_zero_cols=True)

        ## rank on training only then prune local cooc
        ranked_by_lineage = _rank_candidate_alleles(
            ard_train=ard_train,
            train_meta=train_meta,
            sample_col=sample_col,
            top_k_per_lineage=top_k_per_lineage,
            include_negative_alleles=include_negative_alleles,
            top_k_negative_per_lineage=top_k_negative_per_lineage,
        )
        ranked_by_lineage = _prune_ranked_by_lineage_cooc(
            ard_subset=ard_train,
            ranked_by_lineage=ranked_by_lineage,
            cooc_threshold=cooc_threshold,
        )
        del ard_train
        ordered_candidate_records = _round_robin_candidates(
            ranked_by_lineage=ranked_by_lineage,
            max_model_alleles=max_model_alleles,
        )
        if not ordered_candidate_records:
            continue
        ordered_candidates = [record["allele_id"] for record in ordered_candidate_records]
        ordered_target_lineages = {
            record["allele_id"]: record["target_lineage"]
            for record in ordered_candidate_records
        }
        ordered_directions = {
            record["allele_id"]: record["evidence_direction"]
            for record in ordered_candidate_records
        }

        train_alleles = _get_sample_allele_sets(ard_obj, fold.train_ids, alleles=ordered_candidates)
        test_alleles = _get_sample_allele_sets(ard_obj, fold.test_ids, alleles=ordered_candidates)
        train_labels = {sample_id: labels_by_sample[sample_id] for sample_id in fold.train_ids}
        test_labels = pd.Series({sample_id: labels_by_sample[sample_id] for sample_id in fold.test_ids})

        prefix_results = []
        
        ## try every prefix so the final choice can stay small
        for prefix_size in range(1, len(ordered_candidates) + 1):
            selected_alleles = ordered_candidates[:prefix_size]
            forced_target_lineages = {
                allele_id: ordered_target_lineages[allele_id]
                for allele_id in selected_alleles
            }
            fold_model = build_sparse_empirical_model_from_sets(
                sample_alleles=train_alleles,
                labels_by_sample=train_labels,
                selected_alleles=selected_alleles,
                species_id=species_id,
                model_id=model_id,
                smoothing_alpha=smoothing_alpha,
                uniform_priors=uniform_priors,
                forced_target_lineages=forced_target_lineages,
            )
            _score_df, prob_df = score_samples_sparse_tables(test_alleles, fold_model)
            y_pred = prob_df.idxmax(axis=1)
            score = _metric_score(test_labels.loc[prob_df.index], y_pred, prob_df, metric)

            prefix_results.append(
                {
                    "group": fold.group,
                    "num_alleles": prefix_size,
                    "score": float(score),
                    "selected_alleles": list(selected_alleles),
                    "allele_target_lineages": {
                        locus["allele_id"]: locus["target_lineage"]
                        for locus in fold_model["loci"]
                    },
                    "allele_evidence_directions": {
                        allele_id: ordered_directions[allele_id]
                        for allele_id in selected_alleles
                    },
                }
            )

        best_score = max(result["score"] for result in prefix_results)
        eligible = [
            result
            for result in prefix_results
            if result["score"] >= best_score - float(score_tolerance)
        ]
        
        ## within tolerance prefer the smallest panel
        best_result = min(eligible, key=lambda result: (result["num_alleles"], -result["score"]))
        best_result["best_fold_score"] = float(best_score)
        fold_results.append(best_result)

        allele_support.update(best_result["selected_alleles"])
        for allele_id, target_lineage in best_result["allele_target_lineages"].items():
            if allele_id in best_result["selected_alleles"]:
                allele_target_support[allele_id][target_lineage] += 1
                allele_direction_support[allele_id][
                    best_result["allele_evidence_directions"][allele_id]
                ] += 1

    if not fold_results:
        raise ValueError("No LOSO fold produced a valid empirical model.")

    ## stable means selected by enough held out folds
    min_support_count = max(1, math.ceil(float(min_fold_support) * len(fold_results)))
    stable_alleles = [
        allele_id
        for allele_id, support_count in allele_support.items()
        if support_count >= min_support_count
    ]
    
    stable_alleles = _stable_allele_order(stable_alleles, allele_support, allele_target_support)
    
    if not stable_alleles:
        ## fallback keeps the run usable when folds disagree completely
        best_fold = max(fold_results, key=lambda result: result["score"])
        stable_alleles = _stable_allele_order(
            best_fold["selected_alleles"],
            allele_support,
            allele_target_support,
        )

    stable_alleles = _limit_maximum_lineage_alleles(
        stable_alleles,
        allele_target_support=allele_target_support,
        max_alleles_per_lineage=max_alleles_per_lineage,
    )
    stable_alleles = _supplement_minimum_lineage_alleles_from_loso(
        selected_alleles=stable_alleles,
        lineages=sorted(meta["___lineage"].unique()),
        allele_support=allele_support,
        allele_target_support=allele_target_support,
        min_alleles_per_lineage=min_alleles_per_lineage,
    )
    stable_alleles = _limit_maximum_lineage_alleles(
        stable_alleles,
        allele_target_support=allele_target_support,
        max_alleles_per_lineage=max_alleles_per_lineage,
    )

    final_target_lineages = {
        allele_id: target_lineage
        for allele_id, target_lineage in _consensus_target_lineages(allele_target_support).items()
        if allele_id in set(stable_alleles)
    }
    final_evidence_directions = {
        allele_id: direction_counts.most_common(1)[0][0]
        for allele_id, direction_counts in allele_direction_support.items()
        if allele_id in set(stable_alleles) and direction_counts
    }

    full_sample_alleles = _get_sample_allele_sets(
        ard_obj,
        meta[sample_col].tolist(),
        alleles=stable_alleles,
    )
    
    ## final model is refit on all samples with the stable allele set
    final_model = build_sparse_empirical_model_from_sets(
        sample_alleles=full_sample_alleles,
        labels_by_sample=labels_by_sample,
        selected_alleles=stable_alleles,
        species_id=species_id,
        model_id=model_id,
        smoothing_alpha=smoothing_alpha,
        uniform_priors=uniform_priors,
        forced_target_lineages=final_target_lineages,
    )
    for locus in final_model["loci"]:
        allele_id = str(locus["allele_id"])
        locus["evidence_direction"] = str(final_evidence_directions.get(allele_id, "positive"))

    positive_marker_ids = [
        str(locus["allele_id"])
        for locus in final_model["loci"]
        if str(locus.get("evidence_direction", "positive")) == "positive"
    ]
    negative_marker_ids = [
        str(locus["allele_id"])
        for locus in final_model["loci"]
        if str(locus.get("evidence_direction", "positive")) == "negative"
    ]
    final_model["summary"]["fold_count"] = len(fold_results)
    final_model["summary"]["min_alleles_per_lineage"] = int(min_alleles_per_lineage)
    final_model["summary"]["max_alleles_per_lineage"] = int(max_alleles_per_lineage)
    final_model["summary"]["lineage_marker_counts"] = {
        record["lineage_id"]: int(record["marker_count"])
        for record in final_model["lineages"]
    }
    final_model["summary"]["unmet_min_alleles_per_lineage"] = _minimum_lineage_alleles_deficits(
        final_model,
        min_alleles_per_lineage,
    )
    final_model["summary"]["lineage_constraint_source"] = "loso_supported"
    final_model["summary"]["positive_marker_count"] = int(len(positive_marker_ids))
    final_model["summary"]["negative_marker_count"] = int(len(negative_marker_ids))
    final_model["provenance"].update(
        {
            "sample_column": sample_col,
            "lineage_column": lineage_col,
            "group_column": group_col,
            "selection_metric": metric,
            "min_fold_support": float(min_fold_support),
            "score_tolerance": float(score_tolerance),
            "top_k_per_lineage": int(top_k_per_lineage),
            "max_model_alleles": int(max_model_alleles),
            "min_lineage_samples": int(min_lineage_samples),
            "min_alleles_per_lineage": int(min_alleles_per_lineage),
            "max_alleles_per_lineage": int(max_alleles_per_lineage),
            "cooc_threshold": None if cooc_threshold is None else float(cooc_threshold),
            "include_negative_alleles": bool(include_negative_alleles),
            "top_k_negative_per_lineage": (
                None if top_k_negative_per_lineage is None else int(top_k_negative_per_lineage)
            ),
            "lineage_constraint_source": "loso_supported",
            "selection_method": "kullbackleibler",
            "selected_allele_evidence_directions": final_evidence_directions,
            "positive_marker_ids": positive_marker_ids,
            "negative_marker_ids": negative_marker_ids,
            "fold_results": fold_results,
        }
    )
    return final_model, fold_results


def write_model_json(model, output_json):
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_schema_model_json(model, str(output_path))
    return output_path
