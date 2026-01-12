from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from trial_outcome_classifier import (
    EnsembleOutcomeClassifier,
    OutcomeLabel,
    TrialOutcome,
)

# Configuration from env
USE_TRANSFORMER = os.environ.get("USE_TRANSFORMER_CLASSIFIER", "0") == "1"
TRANSFORMER_WEIGHT = float(os.environ.get("TRANSFORMER_WEIGHT", "0.6"))
PATTERN_WEIGHT = float(os.environ.get("PATTERN_WEIGHT", "0.4"))


# =====================================================================
# Helpers
# =====================================================================


def _simplify_for_compatibility(primary_label: str) -> str:
    """
    Convert detailed label to simple met/not_met/unknown for backward compatibility.
    """
    met_labels = {
        "primary_endpoint_met",
        "superiority_shown",
        "clinical_benefit_demonstrated",
        "fda_approval_obtained",
        "secondary_endpoint_met",
        "statistically_significant",
    }

    not_met_labels = {
        "primary_endpoint_not_met",
        "stopped_for_futility",
        "no_clinical_benefit",
        "terminated_early",
        "secondary_endpoint_not_met",
        "not_statistically_significant",
    }

    if primary_label in met_labels:
        return "met"
    elif primary_label in not_met_labels:
        return "not_met"
    else:
        return "unknown"


def _format_evidence(outcome: TrialOutcome) -> str:
    if not outcome.evidence_spans:
        return ""
    parts = []
    for label, spans in list(outcome.evidence_spans.items())[:3]:
        if spans:
            parts.append(f"{label.value}: '{spans[0][:100]}'")
    return " | ".join(parts)


# =====================================================================
# Enhanced PubMed classification
# =====================================================================


def enhanced_pubmed_classification(
    df: pd.DataFrame,
    pubmed_texts: Dict[str, str],
) -> pd.DataFrame:
    """
    Enhanced replacement for the PubMed labeling section.
    Adds multi-label outcomes and keeps compatibility columns.
    """
    print("\n" + "=" * 80)
    print("ENHANCED PUBMED CLASSIFICATION WITH MULTI-LABEL OUTCOMES")
    print("=" * 80)
    print(f"Transformer enabled: {USE_TRANSFORMER}")
    if USE_TRANSFORMER:
        print(f"Weights - Pattern: {PATTERN_WEIGHT}, Transformer: {TRANSFORMER_WEIGHT}")

    classifier = EnsembleOutcomeClassifier(
        use_transformer=USE_TRANSFORMER,
        pattern_weight=PATTERN_WEIGHT,
        transformer_weight=TRANSFORMER_WEIGHT,
    )

    texts = []
    pmids_used = []

    for _, row in df.iterrows():
        pmids = [
            p.strip()
            for p in str(row.get("pubmed_id", "")).split(";")
            if p.strip().isdigit()
        ]
        text = ""
        used_pmid = ""
        for pmid in pmids:
            if pubmed_texts.get(pmid):
                text = pubmed_texts[pmid]
                used_pmid = pmid
                break
        texts.append(text)
        pmids_used.append(used_pmid)

    print(f"Classifying {len(texts)} PubMed abstracts...")
    outcomes: List[TrialOutcome] = []
    for i, text in enumerate(tqdm(texts, desc="PubMed Classification")):
        outcome = classifier.classify(text)
        outcomes.append(outcome)

    # Core columns
    df["pubmed_text_used_pmid"] = pmids_used
    df["outcome_labels_all"] = [
        ";".join([l.value for l in o.labels]) for o in outcomes
    ]
    df["outcome_primary_label"] = [
        o.primary_label.value if o.primary_label else "outcome_unknown"
        for o in outcomes
    ]
    df["outcome_confidence"] = [o.overall_confidence for o in outcomes]
    df["outcome_reasoning"] = [o.reasoning for o in outcomes]
    df["outcome_evidence_spans"] = [_format_evidence(o) for o in outcomes]

    # Boolean flags
    df["has_primary_met"] = [OutcomeLabel.PRIMARY_MET in o.labels for o in outcomes]
    df["has_primary_not_met"] = [
        OutcomeLabel.PRIMARY_NOT_MET in o.labels for o in outcomes
    ]
    df["has_futility"] = [OutcomeLabel.FUTILITY in o.labels for o in outcomes]
    df["has_early_termination"] = [
        OutcomeLabel.EARLY_TERMINATION in o.labels for o in outcomes
    ]
    df["has_safety_concerns"] = [
        OutcomeLabel.SAFETY_CONCERNS in o.labels for o in outcomes
    ]
    df["has_clinical_benefit"] = [
        OutcomeLabel.CLINICAL_BENEFIT in o.labels for o in outcomes
    ]
    df["has_no_clinical_benefit"] = [
        OutcomeLabel.NO_CLINICAL_BENEFIT in o.labels for o in outcomes
    ]
    df["has_significant_result"] = [
        OutcomeLabel.SIGNIFICANT_DIFFERENCE in o.labels for o in outcomes
    ]
    df["has_no_significant_result"] = [
        OutcomeLabel.NO_SIGNIFICANT_DIFFERENCE in o.labels for o in outcomes
    ]

    # Backward compatibility
    df["outcome_simple"] = df["outcome_primary_label"].apply(_simplify_for_compatibility)
    df["pubmed_label_scored"] = df["outcome_simple"]
    df["pubmed_score"] = df["outcome_confidence"] * 10.0
    df["pubmed_evidence_patterns"] = df["outcome_reasoning"]

    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    print("\nPrimary Label Distribution:")
    print(df["outcome_primary_label"].value_counts())

    print("\nSimplified Label Distribution (for compatibility):")
    print(df["outcome_simple"].value_counts())

    print("\nMulti-Label Flags:")
    flag_cols = [c for c in df.columns if c.startswith("has_")]
    for col in flag_cols:
        count = df[col].sum()
        print(f"  {col}: {count} ({(count/len(df)*100 if len(df) else 0):.1f}%)")

    print(f"\nAverage Confidence: {df['outcome_confidence'].mean():.3f}")

    return df


# =====================================================================
# Enhanced fusion with CT.gov (optional)
# =====================================================================


def enhanced_fusion_with_ctgov(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusion that considers multi-label outcomes and CT.gov if present.
    If CT.gov columns are missing, uses PubMed-only signals.
    """
    print("\n" + "=" * 80)
    print("ENHANCED FUSION: PubMed Multi-Label + ClinicalTrials.gov")
    print("=" * 80)

    has_ctgov = "ctgov_failure_score" in df.columns

    final_labels: List[str] = []
    confidence_levels: List[str] = []
    reasoning_list: List[str] = []

    for _, row in df.iterrows():
        primary_label = row.get("outcome_primary_label", "outcome_unknown")
        pubmed_conf = float(row.get("outcome_confidence", 0.0))
        has_futility = bool(row.get("has_futility", False))
        has_early_term = bool(row.get("has_early_termination", False))
        has_safety = bool(row.get("has_safety_concerns", False))

        ct_score = 0
        if has_ctgov:
            try:
                ct_score = int(row.get("ctgov_failure_score", 0))
            except Exception:
                ct_score = 0

        # 1. High confidence primary met / not met
        if primary_label == "primary_endpoint_met" and pubmed_conf >= 0.6:
            final_labels.append("met")
            confidence_levels.append("high")
            reasoning_list.append(
                f"High-confidence PubMed: primary met (conf={pubmed_conf:.2f})"
            )

        elif primary_label == "primary_endpoint_not_met" and pubmed_conf >= 0.6:
            final_labels.append("not_met")
            confidence_levels.append("high")
            reasoning_list.append(
                f"High-confidence PubMed: primary not met (conf={pubmed_conf:.2f})"
            )

        # 2. Futility (PubMed and/or CT.gov)
        elif has_futility or (has_ctgov and ct_score >= 6):
            final_labels.append("not_met_futility")
            confidence_levels.append("high")
            reasoning_list.append(
                f"Futility detected (PubMed futility={has_futility}, CT.gov score={ct_score})"
            )

        # 3. Early termination with negative signal
        elif has_early_term and primary_label in [
            "primary_endpoint_not_met",
            "no_clinical_benefit",
        ]:
            final_labels.append("not_met_early_term")
            confidence_levels.append("medium-high")
            reasoning_list.append("Early termination + negative PubMed signal")

        # 4. Safety-driven
        elif has_safety and pubmed_conf >= 0.5:
            final_labels.append("terminated_safety")
            confidence_levels.append("medium-high")
            reasoning_list.append("Safety concerns detected")

        # 5. Clinical benefit
        elif primary_label == "clinical_benefit_demonstrated" and pubmed_conf >= 0.5:
            final_labels.append("clinical_benefit")
            confidence_levels.append("medium")
            reasoning_list.append("Clinical benefit shown")

        # 6. Medium confidence positives / negatives
        elif primary_label in [
            "primary_endpoint_met",
            "superiority_shown",
            "statistically_significant",
        ] and pubmed_conf >= 0.4:
            final_labels.append("likely_met")
            confidence_levels.append("medium")
            reasoning_list.append(
                f"Medium-confidence positive outcome (conf={pubmed_conf:.2f})"
            )

        elif primary_label in [
            "primary_endpoint_not_met",
            "no_clinical_benefit",
            "not_statistically_significant",
        ] and pubmed_conf >= 0.4:
            final_labels.append("likely_not_met")
            confidence_levels.append("medium")
            reasoning_list.append(
                f"Medium-confidence negative outcome (conf={pubmed_conf:.2f})"
            )

        # 7. CT.gov signal only (if available)
        elif has_ctgov and ct_score >= 2 and primary_label == "outcome_unknown":
            final_labels.append("not_met_candidate")
            confidence_levels.append("low-medium")
            reasoning_list.append("CT.gov terminated/suspended, unclear PubMed signal")

        # 8. Unknown
        else:
            final_labels.append("unknown")
            confidence_levels.append("low")
            reasoning_list.append(
                f"No clear signals (PubMed: {primary_label}, conf={pubmed_conf:.2f}, CT.gov={ct_score})"
            )

    df["final_outcome_label"] = final_labels
    df["final_confidence"] = confidence_levels
    df["final_reasoning"] = reasoning_list

    print("\nFinal Outcome Distribution:")
    print(df["final_outcome_label"].value_counts())

    print("\nConfidence Level Distribution:")
    print(df["final_confidence"].value_counts())

    return df


# =====================================================================
# Failed trials report (robust to missing CT.gov columns)
# =====================================================================


def create_failed_trials_report(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create enhanced failed trials report with multi-label details.
    Works even if CT.gov columns are missing.
    """
    print("\n" + "=" * 80)
    print("CREATING ENHANCED FAILED TRIALS REPORT")
    print("=" * 80)

    has_ctgov_failure_score = "ctgov_failure_score" in df.columns

    # Base filter for failures using PubMed-derived labels and flags
    failure_mask = (
        df["final_outcome_label"].isin(
            [
                "not_met",
                "not_met_futility",
                "not_met_early_term",
                "terminated_safety",
                "likely_not_met",
                "not_met_candidate",
            ]
        )
        | df.get("has_primary_not_met", False)
        | df.get("has_futility", False)
    )

    if has_ctgov_failure_score:
        # Add CT.gov-based failure signal if available
        failure_mask = failure_mask | (df["ctgov_failure_score"] >= 4)

    failed_df = df[failure_mask].copy()

    # Sort by confidence and date if available
    failed_df["outcome_confidence_rank"] = failed_df["outcome_confidence"]
    sort_cols = ["outcome_confidence_rank"]
    sort_asc = [False]

    if "initial_data_release_date" in failed_df.columns:
        sort_cols.append("initial_data_release_date")
        sort_asc.append(False)

    failed_df = failed_df.sort_values(
        by=sort_cols,
        ascending=sort_asc,
    )

    # Columns for report (only keep those that exist)
    report_cols = [
        "study_accession",
        "technology",
        "brief_title",
        "final_outcome_label",
        "final_confidence",
        "outcome_primary_label",
        "outcome_confidence",
        "outcome_labels_all",
        "has_primary_not_met",
        "has_futility",
        "has_safety_concerns",
        "ctgov_failure_score",
        "ctgov_best_status",
        "nct_ids_merged",
        "pubmed_id",
        "outcome_reasoning",
        "final_reasoning",
        "initial_data_release_date",
    ]
    available_cols = [c for c in report_cols if c in failed_df.columns]
    failed_report = failed_df[available_cols]

    out_path = os.path.join(output_dir, "immport_failed_trials_enhanced.csv")
    failed_report.to_csv(out_path, index=False)

    print("\nFailed Trials Report:")
    print(f"  Total failed: {len(failed_report)}")
    print(
        f"  High confidence: "
        f"{len(failed_report[failed_report.get('final_confidence', '') == 'high'])}"
    )
    print(f"  With futility: {int(failed_report.get('has_futility', False).sum())}")
    print(
        f"  With safety concerns: "
        f"{int(failed_report.get('has_safety_concerns', False).sum())}"
    )
    print(f"  Saved to: {out_path}")

    # Summary JSON
    summary_stats = {
        "total_failed_trials": len(failed_report),
        "by_confidence": failed_report.get("final_confidence", pd.Series([]))
        .value_counts()
        .to_dict(),
        "by_outcome_label": failed_report.get("final_outcome_label", pd.Series([]))
        .value_counts()
        .to_dict(),
        "with_futility": int(failed_report.get("has_futility", False).sum()),
        "with_safety_concerns": int(
            failed_report.get("has_safety_concerns", False).sum()
        ),
        "flow_cytometry": int(
            failed_report.get("technology", pd.Series([])).eq("Flow Cytometry").sum()
        ),
        "cytof": int(
            failed_report.get("technology", pd.Series([]))
            .eq("CyTOF/Mass Cytometry")
            .sum()
        ),
    }

    summary_path = os.path.join(output_dir, "failed_trials_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  Summary saved to: {summary_path}")


# =====================================================================
# Integration entry point
# =====================================================================


def integrate_into_pipeline(
    df: pd.DataFrame,
    pubmed_texts: Dict[str, str],
    output_dir: str,
) -> pd.DataFrame:
    """
    Complete integration that replaces the old PubMed labeling + fusion.
    1) Enhanced PubMed classification
    2) Save intermediate CSV
    3) Enhanced fusion (uses CT.gov if available)
    4) Create failed trials report
    """
    # Step 1: PubMed classification
    df = enhanced_pubmed_classification(df, pubmed_texts)

    # Save intermediate
    out_pubmed = os.path.join(
        output_dir, "immport_cytometry_candidates_pubmed_enhanced.csv"
    )
    df.to_csv(out_pubmed, index=False)
    print(f"\nSaved: {out_pubmed}")

    # Step 2: Fusion with CT.gov if columns present
    if "ctgov_failure_score" in df.columns:
        df = enhanced_fusion_with_ctgov(df)
    else:
        print("\nWARNING: CT.gov columns not found. Skipping CT.gov fusion step.")
        # Fallback: final_* derived from PubMed-only
        df["final_outcome_label"] = df["outcome_simple"]
        df["final_confidence"] = [
            "medium" if c > 0.5 else "low" for c in df["outcome_confidence"]
        ]
        df["final_reasoning"] = df["outcome_reasoning"]

    # Step 3: Failed trials report (robust to missing CT.gov)
    create_failed_trials_report(df, output_dir)

    return df


if __name__ == "__main__":
    print("pipeline_integration.py - helper module for ImmPort pipeline")
