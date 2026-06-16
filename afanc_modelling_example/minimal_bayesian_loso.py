import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from model_json_library import build_empirical_loso_model, write_model_json as write_schema_model_json
from minimal_empirical_loso import (
    _consensus_target_lineages,
    _get_sample_allele_sets,
    _limit_maximum_lineage_alleles,
    _make_loso_folds,
    _metric_score,
    _minimum_lineage_alleles_deficits,
    _parse_ardal_allele,
    _rank_candidate_alleles,
    _round_robin_candidates,
    _stable_allele_order,
    _subset_ardal,
    _supplement_minimum_lineage_alleles_from_loso,
    score_samples_sparse_tables,
)


DEFAULT_TARGET_MU_PRIOR = {"alpha": 7.0, "beta": 5.0}
DEFAULT_BACKGROUND_MU_PRIOR = {"alpha": 2.0, "beta": 8.0}
DEFAULT_COOC_THRESHOLD = 0.98


def _resolve_mu_prior(mu_prior, default_prior):
    resolved = dict(default_prior)
    if mu_prior is not None:
        resolved.update(mu_prior)

    ## priors need to be proper beta shapes before any fold work
    alpha = float(resolved["alpha"])
    beta = float(resolved["beta"])
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Beta prior alpha and beta must be positive.")
    return {"alpha": alpha, "beta": beta}


def _beta_posterior_map(k, n, alpha, beta):
    posterior_alpha = float(k) + float(alpha)
    posterior_beta = float(n - k) + float(beta)
    ## use MAP when the posterior has an interior mode
    if posterior_alpha > 1.0 and posterior_beta > 1.0:
        return float((posterior_alpha - 1.0) / (posterior_alpha + posterior_beta - 2.0))
    return float(posterior_alpha / (posterior_alpha + posterior_beta))


def _prune_ranked_by_lineage_cooc(ard_subset, ranked_by_lineage, cooc_threshold, threads=12):
    if cooc_threshold is None:
        return ranked_by_lineage

    pruned = {}
    for lineage, ranked_items in ranked_by_lineage.items():
        lineage_alleles = [str(record["allele_id"]) for record in ranked_items]
        if len(lineage_alleles) < 2:
            pruned[lineage] = [dict(record) for record in ranked_items]
            continue

        ## same cooc thinning as empirical but kept local for this file
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


def build_sparse_bayesian_model_from_sets(
    sample_alleles,
    labels_by_sample,
    selected_alleles,
    species_id="unknown",
    model_id="minimal_bayesian_loso_model",
    uniform_priors=False,
    target_mu_prior=None,
    background_mu_prior=None,
    forced_target_lineages=None,
):
    target_prior = _resolve_mu_prior(target_mu_prior, DEFAULT_TARGET_MU_PRIOR)
    background_prior = _resolve_mu_prior(background_mu_prior, DEFAULT_BACKGROUND_MU_PRIOR)

    ## build per lineage sample buckets for posterior updates
    lineage_to_samples = {}
    for sample_id, lineage in labels_by_sample.items():
        lineage_to_samples.setdefault(str(lineage), set()).add(str(sample_id))

    lineages = sorted(lineage_to_samples)
    all_samples = set(labels_by_sample)
    lineage_counts = {lineage: len(sample_ids) for lineage, sample_ids in lineage_to_samples.items()}
    total_samples = sum(lineage_counts.values())

    if uniform_priors:
        priors = {lineage: 1.0 / len(lineages) for lineage in lineages}
    else:
        priors = {lineage: lineage_counts[lineage] / total_samples for lineage in lineages}

    loci = []
    locus_ids_by_lineage = {lineage: [] for lineage in lineages}

    for allele_id in selected_alleles:
        ## semi bayes swaps smoothed rates for beta posterior estimates
        carriers = {
            sample_id
            for sample_id, allele_set in sample_alleles.items()
            if allele_id in allele_set
        }

        lineage_mus = {}
        for lineage in lineages:
            lineage_ids = lineage_to_samples[lineage]
            n = len(lineage_ids)
            k = len(carriers & lineage_ids)
            lineage_mus[lineage] = _beta_posterior_map(
                k=k,
                n=n,
                alpha=target_prior["alpha"],
                beta=target_prior["beta"],
            )

        target_lineage = None
        if forced_target_lineages is not None:
            ## preserve the LOSO selected lineage during the final refit
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
                key=lambda lineage: (lineage_mus[lineage], lineage),
            )
        background_ids = all_samples - lineage_to_samples[target_lineage]
        background_k = len(carriers & background_ids)
        background_n = len(background_ids)
        background_mu = _beta_posterior_map(
            k=background_k,
            n=background_n,
            alpha=background_prior["alpha"],
            beta=background_prior["beta"],
        )

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
                    "target_frequency": float(lineage_mus[target_lineage]),
                    "background_frequency": float(background_mu),
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
        emission_model="semi_bayes",
        summary={
            "lineage_count": len(lineage_records),
            "selected_locus_count_after_loso": len(loci),
        },
        generation_metadata={
            "builder_function": "build_sparse_bayesian_model_from_sets",
        },
        builder_parameters={
            "uniform_priors": bool(uniform_priors),
            "target_mu_prior": target_prior,
            "background_mu_prior": background_prior,
        },
        provenance={
            "created_by": "build_lineage_min_bayesian_model_sparse",
            "uniform_priors": bool(uniform_priors),
            "target_mu_prior": target_prior,
            "background_mu_prior": background_prior,
        },
    )


def build_lineage_min_bayesian_model_sparse(
    ard_obj,
    meta_df,
    sample_col="Sample",
    lineage_col="Lineage",
    group_col="Country",
    top_k_per_lineage=200,
    max_model_alleles=500,
    min_lineage_samples=10,
    min_fold_support=0.5,
    uniform_priors=False,
    metric="macro_f1",
    species_id="unknown",
    model_id="minimal_bayesian_loso_model",
    score_tolerance=0.005,
    min_alleles_per_lineage=0,
    max_alleles_per_lineage=0,
    target_mu_prior=None,
    background_mu_prior=None,
    cooc_threshold=DEFAULT_COOC_THRESHOLD,
    include_negative_alleles=False,
    top_k_negative_per_lineage=None,
    strict_min_alleles_per_lineage=False,
):
    if int(max_alleles_per_lineage) > 0 and int(min_alleles_per_lineage) > int(max_alleles_per_lineage):
        raise ValueError("max_alleles_per_lineage must be >= min_alleles_per_lineage.")

    ## keep ids as strings before splitting folds
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

        ## rank candidates without seeing the held out group
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
        ## prefix search chooses the smallest panel inside tolerance
        for prefix_size in range(1, len(ordered_candidates) + 1):
            selected_alleles = ordered_candidates[:prefix_size]
            forced_target_lineages = {
                allele_id: ordered_target_lineages[allele_id]
                for allele_id in selected_alleles
            }
            fold_model = build_sparse_bayesian_model_from_sets(
                sample_alleles=train_alleles,
                labels_by_sample=train_labels,
                selected_alleles=selected_alleles,
                species_id=species_id,
                model_id=model_id,
                uniform_priors=uniform_priors,
                target_mu_prior=target_mu_prior,
                background_mu_prior=background_mu_prior,
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
        raise ValueError("No LOSO fold produced a valid bayesian model.")

    ## keep markers that are stable across held out groups
    min_support_count = max(1, math.ceil(float(min_fold_support) * len(fold_results)))
    stable_alleles = [
        allele_id
        for allele_id, support_count in allele_support.items()
        if support_count >= min_support_count
    ]
    stable_alleles = _stable_allele_order(stable_alleles, allele_support, allele_target_support)
    if not stable_alleles:
        ## fallback to best fold if nothing clears the support floor
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
    ## refit once on all samples after LOSO has chosen the allele set
    final_model = build_sparse_bayesian_model_from_sets(
        sample_alleles=full_sample_alleles,
        labels_by_sample=labels_by_sample,
        selected_alleles=stable_alleles,
        species_id=species_id,
        model_id=model_id,
        uniform_priors=uniform_priors,
        target_mu_prior=target_mu_prior,
        background_mu_prior=background_mu_prior,
        forced_target_lineages=final_target_lineages,
    )
    for locus in final_model["loci"]:
        allele_id = str(locus["allele_id"])
        ## direction is selection metadata not part of the emission fit
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
    unmet_lineages = _minimum_lineage_alleles_deficits(final_model, min_alleles_per_lineage)
    final_model["summary"]["unmet_min_alleles_per_lineage"] = unmet_lineages
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
            "strict_min_alleles_per_lineage": bool(strict_min_alleles_per_lineage),
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
    if unmet_lineages and strict_min_alleles_per_lineage:
        ## strict mode turns a soft panel target into a hard failure
        message = ", ".join(
            f"{lineage}={count}" for lineage, count in sorted(unmet_lineages.items())
        )
        raise ValueError(
            f"Could not satisfy min_alleles_per_lineage={int(min_alleles_per_lineage)} for: {message}"
        )
    return final_model, fold_results


def write_model_json(model, output_json):
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_schema_model_json(model, str(output_path))
    return output_path
