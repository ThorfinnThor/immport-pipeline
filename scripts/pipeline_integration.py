
"""
INTEGRATION MODULE - Replace simple regex labeling in immport_cytometry_pipeline_full.py

This module provides drop-in replacements for the existing labeling functions.
"""

from __future__ import annotations
import os
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

# Import the new classifier
from trial_outcome_classifier import (
    EnsembleOutcomeClassifier,
    OutcomeLabel,
    TrialOutcome,
    classify_trial_outcomes_batch
)

# Configuration
USE_TRANSFORMER = os.environ.get("USE_TRANSFORMER_CLASSIFIER", "0") == "1"
TRANSFORMER_WEIGHT = float(os.environ.get("TRANSFORMER_WEIGHT", "0.6"))
PATTERN_WEIGHT = float(os.environ.get("PATTERN_WEIGHT", "0.4"))


# ============================================================================
# DROP-IN REPLACEMENT FUNCTIONS
# ============================================================================

def enhanced_pubmed_classification(
    df: pd.DataFrame,
    pubmed_texts: Dict[str, str]
) -> pd.DataFrame:
    """
    Enhanced replacement for the PubMed labeling section in main().

    REPLACES THIS SECTION in immport_cytometry_pipeline_full.py:

    # -----------------------------
    # 4) PubMed labeling per row (both variants) + extract NCTs from PubMed text
    # -----------------------------

    Args:
        df: DataFrame with 'pubmed_id' column
        pubmed_texts: Dict mapping PMID -> title+abstract text

    Returns:
        Enhanced DataFrame with multi-label classifications
    """
    print(f"\n{'='*80}")
    print("ENHANCED PUBMED CLASSIFICATION WITH MULTI-LABEL OUTCOMES")
    print(f"{'='*80}")
    print(f"Transformer enabled: {USE_TRANSFORMER}")
    if USE_TRANSFORMER:
        print(f"Weights - Pattern: {PATTERN_WEIGHT}, Transformer: {TRANSFORMER_WEIGHT}")

    # Initialize classifier
    classifier = EnsembleOutcomeClassifier(
        use_transformer=USE_TRANSFORMER,
        pattern_weight=PATTERN_WEIGHT,
        transformer_weight=TRANSFORMER_WEIGHT
    )

    # Prepare texts for classification
    texts = []
    pmids_used = []

    for _, row in df.iterrows():
        pmids = [p.strip() for p in str(row["pubmed_id"]).split(";") if p.strip().isdigit()]

        # Find first available text
        text = ""
        used_pmid = ""
        for pmid in pmids:
            if pubmed_texts.get(pmid):
                text = pubmed_texts[pmid]
                used_pmid = pmid
                break

        texts.append(text)
        pmids_used.append(used_pmid)

    # Batch classify
    print(f"Classifying {len(texts)} PubMed abstracts...")
    outcomes = []

    for i, text in enumerate(tqdm(texts, desc="PubMed Classification")):
        outcome = classifier.classify(text)
        outcomes.append(outcome)

    # Add new columns
    df["pubmed_text_used_pmid"] = pmids_used

    # Multi-label outcomes
    df["outcome_labels_all"] = [";".join([l.value for l in o.labels]) for o in outcomes]
    df["outcome_primary_label"] = [o.primary_label.value if o.primary_label else "unknown" for o in outcomes]
    df["outcome_confidence"] = [o.overall_confidence for o in outcomes]
    df["outcome_reasoning"] = [o.reasoning for o in outcomes]

    # Individual label flags (for easy filtering)
    df["has_primary_met"] = [OutcomeLabel.PRIMARY_MET in o.labels for o in outcomes]
    df["has_primary_not_met"] = [OutcomeLabel.PRIMARY_NOT_MET in o.labels for o in outcomes]
    df["has_futility"] = [OutcomeLabel.FUTILITY in o.labels for o in outcomes]
    df["has_early_termination"] = [OutcomeLabel.EARLY_TERMINATION in o.labels for o in outcomes]
    df["has_safety_concerns"] = [OutcomeLabel.SAFETY_CONCERNS in o.labels for o in outcomes]
    df["has_clinical_benefit"] = [OutcomeLabel.CLINICAL_BENEFIT in o.labels for o in outcomes]
    df["has_no_clinical_benefit"] = [OutcomeLabel.NO_CLINICAL_BENEFIT in o.labels for o in outcomes]
    df["has_significant_result"] = [OutcomeLabel.SIGNIFICANT_DIFFERENCE in o.labels for o in outcomes]
    df["has_no_significant_result"] = [OutcomeLabel.NO_SIGNIFICANT_DIFFERENCE in o.labels for o in outcomes]

    # Evidence spans (formatted as JSON-like string)
    def format_evidence(outcome: TrialOutcome) -> str:
        if not outcome.evidence_spans:
            return ""
        parts = []
        for label, spans in list(outcome.evidence_spans.items())[:3]:
            if spans:
                parts.append(f"{label.value}: '{spans[0][:100]}'")
        return " | ".join(parts)

    df["outcome_evidence_spans"] = [format_evidence(o) for o in outcomes]

    # Backward compatibility: Simple labels
    df["outcome_simple"] = df["outcome_primary_label"].apply(_simplify_for_compatibility)

    # Legacy columns (for compatibility with existing code)
    df["pubmed_label_scored"] = df["outcome_simple"]  # met/not_met/unknown
    df["pubmed_score"] = df["outcome_confidence"] * 10  # Scale to 0-10
    df["pubmed_evidence_patterns"] = df["outcome_reasoning"]

    # Summary statistics
    print(f"\n{'='*80}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nPrimary Label Distribution:")
    print(df["outcome_primary_label"].value_counts())

    print(f"\nSimplified Label Distribution (for compatibility):")
    print(df["outcome_simple"].value_counts())

    print(f"\nMulti-Label Flags:")
    flag_cols = [c for c in df.columns if c.startswith("has_")]
    for col in flag_cols:
        count = df[col].sum()
        print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")

    print(f"\nAverage Confidence: {df['outcome_confidence'].mean():.3f}")

    return df


def _simplify_for_compatibility(primary_label: str) -> str:
    """Convert detailed label to simple met/not_met/unknown for backward compatibility."""
    met_labels = {
        "primary_endpoint_met", 
        "superiority_shown", 
        "clinical_benefit_demonstrated", 
        "fda_approval_obtained",
        "secondary_endpoint_met"
    }

    not_met_labels = {
        "primary_endpoint_not_met", 
        "stopped_for_futility",
        "no_clinical_benefit", 
        "terminated_early",
        "secondary_endpoint_not_met",
        "not_statistically_significant"
    }

    if primary_label in met_labels:
        return "met"
    elif primary_label in not_met_labels:
        return "not_met"
    else:
        return "unknown"


def enhanced_fusion_with_ctgov(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Enhanced fusion that considers multi-label outcomes.

    REPLACES the fusion section in immport_cytometry_pipeline_full.py:
    # -----------------------------
    # 6) Fusion
    # -----------------------------

    Args:
        df: DataFrame with outcome_primary_label and ctgov_failure_score columns

    Returns:
        DataFrame with enhanced final labels
    """
    print(f"\n{'='*80}")
    print("ENHANCED FUSION: PubMed Multi-Label + ClinicalTrials.gov")
    print(f"{'='*80}")

    final_labels = []
    confidence_levels = []
    reasoning_list = []

    for _, row in df.iterrows():
        primary_label = row.get("outcome_primary_label", "unknown")
        pubmed_conf = row.get("outcome_confidence", 0.0)
        ct_score = row.get("ctgov_failure_score", 0)
        has_futility = row.get("has_futility", False)
        has_early_term = row.get("has_early_termination", False)

        # Decision tree with multi-label awareness

        # 1. High confidence PubMed primary endpoint results
        if primary_label == "primary_endpoint_met" and pubmed_conf >= 0.6:
            final_labels.append("met")
            confidence_levels.append("high")
            reasoning_list.append(f"High-confidence PubMed: primary met (conf={pubmed_conf:.2f})")

        elif primary_label == "primary_endpoint_not_met" and pubmed_conf >= 0.6:
            final_labels.append("not_met")
            confidence_levels.append("high")
            reasoning_list.append(f"High-confidence PubMed: primary not met (conf={pubmed_conf:.2f})")

        # 2. Futility signal (very strong)
        elif has_futility or (ct_score >= 6):
            final_labels.append("not_met_futility")
            confidence_levels.append("high")
            reasoning_list.append(f"Futility detected (PubMed futility={has_futility}, CT.gov score={ct_score})")

        # 3. Early termination with moderate PubMed evidence
        elif has_early_term and primary_label in ["primary_endpoint_not_met", "no_clinical_benefit"]:
            final_labels.append("not_met_early_term")
            confidence_levels.append("medium-high")
            reasoning_list.append(f"Early termination + negative PubMed signal")

        # 4. Clinical benefit (even if primary not formally met)
        elif primary_label == "clinical_benefit_demonstrated" and pubmed_conf >= 0.5:
            final_labels.append("clinical_benefit")
            confidence_levels.append("medium")
            reasoning_list.append(f"Clinical benefit shown despite primary endpoint status")

        # 5. Safety concerns dominate
        elif row.get("has_safety_concerns", False) and pubmed_conf >= 0.5:
            final_labels.append("terminated_safety")
            confidence_levels.append("medium-high")
            reasoning_list.append(f"Safety concerns detected")

        # 6. Medium confidence results
        elif primary_label in ["primary_endpoint_met", "superiority_shown"] and pubmed_conf >= 0.4:
            final_labels.append("likely_met")
            confidence_levels.append("medium")
            reasoning_list.append(f"Medium-confidence positive outcome (conf={pubmed_conf:.2f})")

        elif primary_label in ["primary_endpoint_not_met", "no_clinical_benefit"] and pubmed_conf >= 0.4:
            final_labels.append("likely_not_met")
            confidence_levels.append("medium")
            reasoning_list.append(f"Medium-confidence negative outcome (conf={pubmed_conf:.2f})")

        # 7. CT.gov terminated/suspended (without clear PubMed signal)
        elif ct_score >= 2 and primary_label == "outcome_unknown":
            final_labels.append("not_met_candidate")
            confidence_levels.append("low-medium")
            reasoning_list.append(f"CT.gov terminated/suspended, unclear PubMed signal")

        # 8. Unknown
        else:
            final_labels.append("unknown")
            confidence_levels.append("low")
            reasoning_list.append(f"No clear signals (PubMed: {primary_label}, conf={pubmed_conf:.2f}, CT.gov={ct_score})")

    df["final_outcome_label"] = final_labels
    df["final_confidence"] = confidence_levels
    df["final_reasoning"] = reasoning_list

    # Summary
    print(f"\nFinal Outcome Distribution:")
    print(df["final_outcome_label"].value_counts())

    print(f"\nConfidence Level Distribution:")
    print(df["final_confidence"].value_counts())

    return df


def create_failed_trials_report(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create enhanced failed trials report with multi-label details.

    ENHANCES the existing output with more granular information.
    """
    print(f"\n{'='*80}")
    print("CREATING ENHANCED FAILED TRIALS REPORT")
    print(f"{'='*80}")

    # Filter for failures (multiple criteria)
    failed_df = df[
        (df["final_outcome_label"].isin([
            "not_met", "not_met_futility", "not_met_early_term", 
            "terminated_safety", "likely_not_met"
        ])) |
        (df["has_primary_not_met"]) |
        (df["has_futility"]) |
        (df["ctgov_failure_score"] >= 4)
    ].copy()

    # Sort by confidence and date
    failed_df["outcome_confidence_rank"] = failed_df["outcome_confidence"]
    failed_df = failed_df.sort_values(
        by=["outcome_confidence_rank", "initial_data_release_date"],
        ascending=[False, False]
    )

    # Select columns for report
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

    # Save
    out_path = os.path.join(output_dir, "immport_failed_trials_enhanced.csv")
    failed_report.to_csv(out_path, index=False)

    print(f"\nFailed Trials Report:")
    print(f"  Total failed: {len(failed_report)}")
    print(f"  High confidence: {len(failed_report[failed_report['final_confidence'] == 'high'])}")
    print(f"  With futility: {failed_report['has_futility'].sum()}")
    print(f"  With safety concerns: {failed_report['has_safety_concerns'].sum()}")
    print(f"  Saved to: {out_path}")

    # Also create a summary
    summary_stats = {
        "total_failed_trials": len(failed_report),
        "by_confidence": failed_report["final_confidence"].value_counts().to_dict(),
        "by_outcome_label": failed_report["final_outcome_label"].value_counts().to_dict(),
        "with_futility": int(failed_report["has_futility"].sum()),
        "with_safety_concerns": int(failed_report["has_safety_concerns"].sum()),
        "flow_cytometry": len(failed_report[failed_report["technology"] == "Flow Cytometry"]),
        "cytof": len(failed_report[failed_report["technology"] == "CyTOF/Mass Cytometry"]),
    }

    import json
    summary_path = os.path.join(output_dir, "failed_trials_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"  Summary saved to: {summary_path}")


# ============================================================================
# MAIN INTEGRATION FUNCTION
# ============================================================================

def integrate_into_pipeline(
    df: pd.DataFrame,
    pubmed_texts: Dict[str, str],
    output_dir: str
) -> pd.DataFrame:
    """
    Complete integration that replaces the labeling sections of the original pipeline.

    Call this INSTEAD OF the original PubMed labeling + fusion sections.

    Usage in immport_cytometry_pipeline_full.py:

        # After fetching PubMed texts (section 3), REPLACE sections 4-6 with:
        from pipeline_integration import integrate_into_pipeline

        df = integrate_into_pipeline(df, pub_texts, OUTPUT_DIR)

        # Then continue with section 7 (outputs)

    Args:
        df: DataFrame after ImmPort search + PubMed fetch
        pubmed_texts: Dict of PMID -> text from efetch_pubmed_title_abstract()
        output_dir: Output directory path

    Returns:
        Enhanced DataFrame with multi-label classifications
    """
    # Step 1: Enhanced PubMed classification
    df = enhanced_pubmed_classification(df, pubmed_texts)

    # Save intermediate output
    out_pubmed = os.path.join(output_dir, "immport_cytometry_candidates_pubmed_enhanced.csv")
    df.to_csv(out_pubmed, index=False)
    print(f"\nSaved: {out_pubmed}")

    # Step 2: Enhanced fusion with CT.gov
    # Note: Assumes CT.gov columns already exist (ctgov_failure_score, etc.)
    # If not, you need to run the CT.gov section first
    if "ctgov_failure_score" in df.columns:
        df = enhanced_fusion_with_ctgov(df)
    else:
        print("\nWARNING: CT.gov columns not found. Skipping fusion.")
        df["final_outcome_label"] = df["outcome_simple"]
        df["final_confidence"] = ["medium" if c > 0.5 else "low" for c in df["outcome_confidence"]]
        df["final_reasoning"] = df["outcome_reasoning"]

    # Step 3: Create failed trials report
    create_failed_trials_report(df, output_dir)

    return df


# ============================================================================
# EXAMPLE: How to modify main() in immport_cytometry_pipeline_full.py
# ============================================================================

def example_main_modification():
    """
    This shows how to modify the main() function in immport_cytometry_pipeline_full.py

    BEFORE (lines ~XXX-XXX):
    ```
    # -----------------------------
    # 4) PubMed labeling per row (both variants) + extract NCTs from PubMed text
    # -----------------------------
    labels_v1, evidence_v1, ncts_from_pubmed = [], [], []
    labels_scored, scores_scored, evidence_scored = [], [], []
    ...

    # -----------------------------
    # 5) CT.gov scoring (best-of up to 5 NCTs)
    # -----------------------------
    ...

    # -----------------------------
    # 6) Fusion
    # -----------------------------
    ...
    ```

    AFTER:
    ```
    # Import the integration module
    from pipeline_integration import integrate_into_pipeline

    # Replace sections 4-6 with one call:
    df = integrate_into_pipeline(df, pub_texts, OUTPUT_DIR)

    # CT.gov scoring still happens (if columns exist), or add here:
    # ... existing CT.gov code from section 5 ...
    ```
    """
    pass


if __name__ == "__main__":
    print("Pipeline Integration Module")
    print("="*80)
    print("This module provides enhanced classification to replace sections 4-6")
    print("of immport_cytometry_pipeline_full.py")
    print()
    print("See example_main_modification() for integration instructions.")

