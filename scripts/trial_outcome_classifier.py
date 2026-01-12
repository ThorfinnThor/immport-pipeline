
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

# Core NLP
try:
    import spacy
    from spacy.language import Language
except ImportError:
    print("WARNING: spacy not installed. Run: pip install spacy")
    print("Then: python -m spacy download en_core_web_sm")

# Transformer-based classification (optional but recommended)
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: transformers not installed. Run: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# OUTCOME TAXONOMY (Multi-label Classification)
# ============================================================================

class OutcomeLabel(Enum):
    """Multi-label outcome categories for clinical trials."""

    # Primary endpoint outcomes
    PRIMARY_MET = "primary_endpoint_met"
    PRIMARY_NOT_MET = "primary_endpoint_not_met"
    PRIMARY_PARTIAL = "primary_endpoint_partially_met"

    # Secondary outcomes
    SECONDARY_MET = "secondary_endpoint_met"
    SECONDARY_NOT_MET = "secondary_endpoint_not_met"

    # Safety outcomes
    SAFETY_CONCERNS = "safety_concerns"
    ADVERSE_EVENTS = "serious_adverse_events"
    TOXICITY = "dose_limiting_toxicity"

    # Trial status
    EARLY_TERMINATION = "terminated_early"
    FUTILITY = "stopped_for_futility"
    COMPLETED = "completed_as_planned"

    # Statistical outcomes
    SIGNIFICANT_DIFFERENCE = "statistically_significant"
    NO_SIGNIFICANT_DIFFERENCE = "not_statistically_significant"
    NON_INFERIORITY = "non_inferiority_shown"
    SUPERIORITY = "superiority_shown"

    # Clinical relevance
    CLINICAL_BENEFIT = "clinical_benefit_demonstrated"
    NO_CLINICAL_BENEFIT = "no_clinical_benefit"

    # Regulatory
    FDA_APPROVED = "fda_approval_obtained"
    REGULATORY_REJECTION = "regulatory_rejection"

    # Unknown
    UNKNOWN = "outcome_unknown"


@dataclass
class TrialOutcome:
    """Structured representation of trial outcomes."""
    labels: List[OutcomeLabel] = field(default_factory=list)
    confidence_scores: Dict[OutcomeLabel, float] = field(default_factory=dict)
    evidence_spans: Dict[OutcomeLabel, List[str]] = field(default_factory=dict)
    primary_label: Optional[OutcomeLabel] = None
    overall_confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": [label.value for label in self.labels],
            "primary_label": self.primary_label.value if self.primary_label else "unknown",
            "confidence_scores": {k.value: v for k, v in self.confidence_scores.items()},
            "evidence_spans": {k.value: v for k, v in self.evidence_spans.items()},
            "overall_confidence": self.overall_confidence,
            "reasoning": self.reasoning
        }


# ============================================================================
# PATTERN-BASED EXTRACTOR (Improved with spaCy)
# ============================================================================

class SpaCyPatternExtractor:
    """
    Advanced pattern matching using spaCy for linguistic features.
    More sophisticated than pure regex.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Downloading spaCy model: {model}")
            os.system(f"python -m spacy download {model}")
            self.nlp = spacy.load(model)

        # Define pattern rules with weights and context
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[OutcomeLabel, List[Dict]]:
        """
        Define sophisticated patterns with context awareness.
        Each pattern includes: text pattern, weight, required context, exclusions
        """
        return {
            # PRIMARY ENDPOINT - MET
            OutcomeLabel.PRIMARY_MET: [
                {
                    "patterns": [
                        r"(?:primary|main|principal)\s+(?:end\s?point|endpoint|outcome)\s+(?:was|were)?\s*(?:met|achieved|reached|attained)",
                        r"met\s+(?:the|its|all)?\s*primary\s+(?:end\s?point|endpoint|outcome)",
                        r"successfully\s+(?:met|achieved).*primary\s+(?:end\s?point|endpoint|outcome)",
                        r"primary\s+(?:end\s?point|endpoint|outcome).*\bmet\b",
                    ],
                    "weight": 5.0,
                    "requires_near": ["significant", "p\s*[<≤]\s*0\.05", "efficacy"],
                    "excludes_near": ["not", "did not", "failed", "unable"],
                    "context_window": 50
                },
                {
                    "patterns": [
                        r"demonstrated\s+(?:significant|substantial)\s+(?:improvement|benefit|efficacy).*primary",
                        r"primary.*showed\s+significant\s+(?:improvement|benefit)",
                    ],
                    "weight": 4.0,
                    "requires_near": [],
                    "excludes_near": ["no", "not", "lack of"],
                    "context_window": 30
                }
            ],

            # PRIMARY ENDPOINT - NOT MET
            OutcomeLabel.PRIMARY_NOT_MET: [
                {
                    "patterns": [
                        r"(?:did not|failed to|unable to)\s+(?:meet|achieve|reach).*primary\s+(?:end\s?point|endpoint|outcome)",
                        r"primary\s+(?:end\s?point|endpoint|outcome)\s+(?:was|were)?\s*(?:not met|unmet|missed)",
                        r"primary\s+(?:end\s?point|endpoint).*(?:not achieved|not reached)",
                        r"failed.*primary\s+(?:end\s?point|endpoint)",
                    ],
                    "weight": 5.0,
                    "requires_near": [],
                    "excludes_near": ["despite", "although", "however.*met"],
                    "context_window": 50
                },
                {
                    "patterns": [
                        r"no\s+significant\s+(?:difference|improvement|benefit|effect).*primary",
                        r"primary.*no\s+significant",
                    ],
                    "weight": 4.0,
                    "requires_near": ["p\s*[>≥]\s*0\.05", "p\s*=\s*(?:0\.[1-9]|1\.0)"],
                    "excludes_near": [],
                    "context_window": 40
                }
            ],

            # FUTILITY / EARLY TERMINATION
            OutcomeLabel.FUTILITY: [
                {
                    "patterns": [
                        r"(?:stopped|terminated|halted|discontinued).*(?:futility|lack of efficacy|insufficient efficacy)",
                        r"futility\s+(?:analysis|boundary|stopping rule)",
                        r"unlikely to demonstrate\s+(?:efficacy|benefit)",
                        r"conditional power.*below.*threshold",
                    ],
                    "weight": 5.0,
                    "requires_near": [],
                    "excludes_near": [],
                    "context_window": 60
                }
            ],

            # EARLY TERMINATION (general)
            OutcomeLabel.EARLY_TERMINATION: [
                {
                    "patterns": [
                        r"(?:prematurely|early)\s+(?:terminated|stopped|discontinued|halted)",
                        r"(?:terminated|stopped|discontinued|halted)\s+(?:early|prematurely|ahead of schedule)",
                        r"trial.*(?:terminated|stopped).*(?:enrollment|recruitment)",
                    ],
                    "weight": 4.0,
                    "requires_near": [],
                    "excludes_near": ["successfully completed", "as planned"],
                    "context_window": 50
                }
            ],

            # SAFETY CONCERNS
            OutcomeLabel.SAFETY_CONCERNS: [
                {
                    "patterns": [
                        r"safety\s+concerns?",
                        r"unacceptable\s+(?:safety|toxicity|adverse events)",
                        r"(?:significant|serious)\s+safety\s+(?:issues?|concerns?|signals?)",
                        r"safety.*(?:stopped|terminated|halted)",
                    ],
                    "weight": 4.5,
                    "requires_near": [],
                    "excludes_near": ["no safety", "acceptable safety", "manageable"],
                    "context_window": 40
                }
            ],

            # ADVERSE EVENTS
            OutcomeLabel.ADVERSE_EVENTS: [
                {
                    "patterns": [
                        r"serious adverse events?\s+\(SAEs?\)",
                        r"SAEs?.*(?:higher|increased|elevated|greater)",
                        r"grade\s+[34]\s+adverse\s+events?",
                        r"\b(?:death|mortality).*(?:treatment|drug).*related",
                    ],
                    "weight": 4.0,
                    "requires_near": [],
                    "excludes_near": ["no serious", "no increase"],
                    "context_window": 50
                }
            ],

            # STATISTICAL SIGNIFICANCE
            OutcomeLabel.SIGNIFICANT_DIFFERENCE: [
                {
                    "patterns": [
                        r"(?:statistically\s+)?significant(?:ly)?.*(?:difference|improvement|reduction|increase)",
                        r"p\s*[<≤]\s*0\.0(?:5|1)",
                        r"95%\s+(?:confidence interval|CI).*(?:exclude[sd]?\s+(?:zero|null)|does not (?:cross|include) (?:zero|null))",
                    ],
                    "weight": 3.5,
                    "requires_near": [],
                    "excludes_near": ["not statistically", "no significant"],
                    "context_window": 40
                }
            ],

            # NO SIGNIFICANT DIFFERENCE
            OutcomeLabel.NO_SIGNIFICANT_DIFFERENCE: [
                {
                    "patterns": [
                        r"no\s+(?:statistically\s+)?significant\s+(?:difference|improvement|effect|benefit)",
                        r"not\s+(?:statistically\s+)?significant(?:ly)?\s+different",
                        r"p\s*[>≥]\s*0\.05",
                        r"95%\s+(?:confidence interval|CI).*(?:include[sd]?|cross(?:es|ed)?).*(?:zero|null)",
                    ],
                    "weight": 4.0,
                    "requires_near": [],
                    "excludes_near": ["despite", "although"],
                    "context_window": 40
                }
            ],

            # SUPERIORITY
            OutcomeLabel.SUPERIORITY: [
                {
                    "patterns": [
                        r"superior(?:ity)?\s+(?:to|over|versus|vs\.?)\s+(?:placebo|control|standard)",
                        r"significantly\s+(?:better|superior)\s+than",
                        r"outperformed\s+(?:placebo|control|comparator)",
                    ],
                    "weight": 4.0,
                    "requires_near": [],
                    "excludes_near": ["not superior", "failed to show superior"],
                    "context_window": 40
                }
            ],

            # NON-INFERIORITY
            OutcomeLabel.NON_INFERIORITY: [
                {
                    "patterns": [
                        r"non-inferior(?:ity)?.*(?:demonstrated|shown|established)",
                        r"met.*non-inferior(?:ity)?",
                        r"non-inferior(?:ity)?.*margin",
                    ],
                    "weight": 4.0,
                    "requires_near": [],
                    "excludes_near": ["did not meet non-inferior", "failed non-inferior"],
                    "context_window": 40
                }
            ],

            # CLINICAL BENEFIT
            OutcomeLabel.CLINICAL_BENEFIT: [
                {
                    "patterns": [
                        r"clinical(?:ly)?\s+(?:meaningful|significant|relevant)\s+(?:benefit|improvement)",
                        r"(?:meaningful|substantial)\s+clinical\s+(?:benefit|efficacy)",
                        r"improved.*(?:survival|quality of life|disease control)",
                    ],
                    "weight": 3.5,
                    "requires_near": [],
                    "excludes_near": ["no clinical", "lack of clinical", "without clinical"],
                    "context_window": 40
                }
            ],

            # NO CLINICAL BENEFIT
            OutcomeLabel.NO_CLINICAL_BENEFIT: [
                {
                    "patterns": [
                        r"no\s+clinical(?:ly)?\s+(?:meaningful|significant|relevant)\s+(?:benefit|improvement)",
                        r"lack\s+of\s+clinical\s+(?:benefit|efficacy)",
                        r"(?:failed|unable)\s+to\s+(?:improve|demonstrate).*clinical",
                    ],
                    "weight": 4.0,
                    "requires_near": [],
                    "excludes_near": [],
                    "context_window": 40
                }
            ],

            # REGULATORY APPROVAL
            OutcomeLabel.FDA_APPROVED: [
                {
                    "patterns": [
                        r"FDA\s+(?:approval|approved)",
                        r"approved\s+by\s+(?:the\s+)?FDA",
                        r"regulatory\s+approval.*(?:granted|obtained|received)",
                        r"marketing\s+authorization.*(?:granted|approved)",
                    ],
                    "weight": 5.0,
                    "requires_near": [],
                    "excludes_near": ["denied", "rejected", "pending", "seeking"],
                    "context_window": 50
                }
            ],
        }

    def extract(self, text: str) -> TrialOutcome:
        """
        Extract outcome labels from text using sophisticated pattern matching.
        """
        if not text or len(text.strip()) < 10:
            return TrialOutcome(
                labels=[OutcomeLabel.UNKNOWN],
                primary_label=OutcomeLabel.UNKNOWN,
                overall_confidence=0.0
            )

        # Process with spaCy
        doc = self.nlp(text.lower())
        text_lower = text.lower()

        # Store matches
        matches: Dict[OutcomeLabel, List[Tuple[float, str]]] = {}

        # Apply patterns
        for label, pattern_groups in self.patterns.items():
            for group in pattern_groups:
                for pattern in group["patterns"]:
                    for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                        # Extract context window
                        start = max(0, match.start() - group["context_window"])
                        end = min(len(text_lower), match.end() + group["context_window"])
                        context = text_lower[start:end]

                        # Check exclusions
                        excluded = False
                        for exclusion in group["excludes_near"]:
                            if re.search(exclusion, context):
                                excluded = True
                                break

                        if excluded:
                            continue

                        # Check required terms
                        if group["requires_near"]:
                            required_found = False
                            for required in group["requires_near"]:
                                if re.search(required, context):
                                    required_found = True
                                    break
                            if not required_found:
                                continue

                        # Valid match
                        evidence = text[match.start():match.end()]
                        weight = group["weight"]

                        if label not in matches:
                            matches[label] = []
                        matches[label].append((weight, evidence))

        # Aggregate scores
        outcome = TrialOutcome()

        for label, match_list in matches.items():
            # Sum weights with diminishing returns
            weights = [w for w, _ in match_list]
            evidence = [e for _, e in match_list]

            # Diminishing returns: first match = full weight, subsequent = 50%
            if weights:
                score = weights[0] + sum(w * 0.5 for w in weights[1:])
            else:
                score = 0.0

            # Normalize to 0-1
            confidence = min(1.0, score / 10.0)

            if confidence > 0.3:  # Threshold for inclusion
                outcome.labels.append(label)
                outcome.confidence_scores[label] = confidence
                outcome.evidence_spans[label] = evidence[:3]  # Top 3 evidence spans

        # Determine primary label
        if outcome.labels:
            # Priority order for primary label
            priority = [
                OutcomeLabel.PRIMARY_MET,
                OutcomeLabel.PRIMARY_NOT_MET,
                OutcomeLabel.FUTILITY,
                OutcomeLabel.EARLY_TERMINATION,
                OutcomeLabel.NO_CLINICAL_BENEFIT,
                OutcomeLabel.CLINICAL_BENEFIT,
                OutcomeLabel.FDA_APPROVED,
            ]

            for prio_label in priority:
                if prio_label in outcome.labels:
                    outcome.primary_label = prio_label
                    outcome.overall_confidence = outcome.confidence_scores[prio_label]
                    break

            # If no priority match, use highest confidence
            if not outcome.primary_label:
                outcome.primary_label = max(
                    outcome.confidence_scores.items(),
                    key=lambda x: x[1]
                )[0]
                outcome.overall_confidence = outcome.confidence_scores[outcome.primary_label]
        else:
            outcome.labels = [OutcomeLabel.UNKNOWN]
            outcome.primary_label = OutcomeLabel.UNKNOWN
            outcome.overall_confidence = 0.0

        # Generate reasoning
        outcome.reasoning = self._generate_reasoning(outcome)

        return outcome

    def _generate_reasoning(self, outcome: TrialOutcome) -> str:
        """Generate human-readable reasoning for the classification."""
        if outcome.primary_label == OutcomeLabel.UNKNOWN:
            return "No clear outcome signals detected in text."

        parts = [
            f"Primary: {outcome.primary_label.value} (conf: {outcome.overall_confidence:.2f})"
        ]

        if len(outcome.labels) > 1:
            other_labels = [l.value for l in outcome.labels if l != outcome.primary_label]
            parts.append(f"Additional: {', '.join(other_labels[:3])}")

        return " | ".join(parts)


# ============================================================================
# TRANSFORMER-BASED CLASSIFIER (BioBERT/PubMedBERT)
# ============================================================================

class TransformerOutcomeClassifier:
    """
    Use pre-trained biomedical transformers for outcome classification.
    This provides state-of-the-art accuracy but requires more compute.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        use_zero_shot: bool = True
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers not available. Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.use_zero_shot = use_zero_shot

        if use_zero_shot:
            # Zero-shot classification (no fine-tuning needed)
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",  # Best zero-shot model
                device=-1  # CPU, use device=0 for GPU
            )
        else:
            # Load BioBERT for sequence classification (would need fine-tuning)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Note: You would need to fine-tune this on labeled trial data
            # For now, we'll use zero-shot as it's more practical
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )

    def classify(self, text: str, max_length: int = 1024) -> TrialOutcome:
        """
        Classify trial outcome using transformer model.
        """
        if not text or len(text.strip()) < 10:
            return TrialOutcome(
                labels=[OutcomeLabel.UNKNOWN],
                primary_label=OutcomeLabel.UNKNOWN,
                overall_confidence=0.0
            )

        # Truncate if too long
        if len(text) > max_length * 4:  # Rough char estimate
            text = text[:max_length * 4]

        # Define candidate labels (natural language)
        candidate_labels = [
            "primary endpoint was met successfully",
            "primary endpoint was not met or failed",
            "trial stopped early for futility or lack of efficacy",
            "serious safety concerns or adverse events",
            "statistically significant positive results",
            "no statistically significant difference found",
            "clinically meaningful benefit demonstrated",
            "no clinical benefit observed",
            "trial terminated early",
            "FDA approval obtained",
        ]

        # Map to our enum
        label_mapping = {
            "primary endpoint was met successfully": OutcomeLabel.PRIMARY_MET,
            "primary endpoint was not met or failed": OutcomeLabel.PRIMARY_NOT_MET,
            "trial stopped early for futility or lack of efficacy": OutcomeLabel.FUTILITY,
            "serious safety concerns or adverse events": OutcomeLabel.SAFETY_CONCERNS,
            "statistically significant positive results": OutcomeLabel.SIGNIFICANT_DIFFERENCE,
            "no statistically significant difference found": OutcomeLabel.NO_SIGNIFICANT_DIFFERENCE,
            "clinically meaningful benefit demonstrated": OutcomeLabel.CLINICAL_BENEFIT,
            "no clinical benefit observed": OutcomeLabel.NO_CLINICAL_BENEFIT,
            "trial terminated early": OutcomeLabel.EARLY_TERMINATION,
            "FDA approval obtained": OutcomeLabel.FDA_APPROVED,
        }

        try:
            # Run classification
            result = self.classifier(
                text,
                candidate_labels,
                multi_label=True,  # Allow multiple labels
                hypothesis_template="This clinical trial outcome indicates that {}."
            )

            outcome = TrialOutcome()

            # Extract labels above threshold
            threshold = 0.3
            for label, score in zip(result["labels"], result["scores"]):
                if score > threshold:
                    enum_label = label_mapping[label]
                    outcome.labels.append(enum_label)
                    outcome.confidence_scores[enum_label] = float(score)

            # Primary label is highest scoring
            if outcome.labels:
                outcome.primary_label = outcome.labels[0]
                outcome.overall_confidence = outcome.confidence_scores[outcome.primary_label]
            else:
                outcome.labels = [OutcomeLabel.UNKNOWN]
                outcome.primary_label = OutcomeLabel.UNKNOWN
                outcome.overall_confidence = 0.0

            outcome.reasoning = f"Transformer classification (top: {outcome.primary_label.value}, conf: {outcome.overall_confidence:.2f})"

            return outcome

        except Exception as e:
            print(f"Transformer classification failed: {e}")
            return TrialOutcome(
                labels=[OutcomeLabel.UNKNOWN],
                primary_label=OutcomeLabel.UNKNOWN,
                overall_confidence=0.0,
                reasoning=f"Classification error: {str(e)}"
            )


# ============================================================================
# ENSEMBLE CLASSIFIER (Combines Pattern + Transformer)
# ============================================================================

class EnsembleOutcomeClassifier:
    """
    Combines pattern-based and transformer-based approaches for best results.
    """

    def __init__(
        self,
        use_transformer: bool = True,
        pattern_weight: float = 0.4,
        transformer_weight: float = 0.6
    ):
        self.pattern_extractor = SpaCyPatternExtractor()
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.pattern_weight = pattern_weight
        self.transformer_weight = transformer_weight

        if self.use_transformer:
            try:
                self.transformer_classifier = TransformerOutcomeClassifier()
            except Exception as e:
                print(f"Failed to load transformer: {e}")
                self.use_transformer = False

    def classify(self, text: str) -> TrialOutcome:
        """
        Classify using ensemble of methods.
        """
        # Get pattern-based results
        pattern_outcome = self.pattern_extractor.extract(text)

        # If transformer not available, return pattern results
        if not self.use_transformer:
            return pattern_outcome

        # Get transformer results
        transformer_outcome = self.transformer_classifier.classify(text)

        # Ensemble: combine scores
        combined_scores: Dict[OutcomeLabel, float] = {}
        all_labels = set(pattern_outcome.labels + transformer_outcome.labels)

        for label in all_labels:
            pattern_score = pattern_outcome.confidence_scores.get(label, 0.0)
            transformer_score = transformer_outcome.confidence_scores.get(label, 0.0)

            combined_score = (
                self.pattern_weight * pattern_score +
                self.transformer_weight * transformer_score
            )
            combined_scores[label] = combined_score

        # Build ensemble outcome
        outcome = TrialOutcome()
        threshold = 0.25

        for label, score in combined_scores.items():
            if score > threshold:
                outcome.labels.append(label)
                outcome.confidence_scores[label] = score

                # Combine evidence from both sources
                evidence = []
                if label in pattern_outcome.evidence_spans:
                    evidence.extend(pattern_outcome.evidence_spans[label])
                outcome.evidence_spans[label] = evidence[:3]

        # Sort labels by confidence
        outcome.labels.sort(key=lambda l: outcome.confidence_scores[l], reverse=True)

        if outcome.labels:
            outcome.primary_label = outcome.labels[0]
            outcome.overall_confidence = outcome.confidence_scores[outcome.primary_label]
        else:
            outcome.labels = [OutcomeLabel.UNKNOWN]
            outcome.primary_label = OutcomeLabel.UNKNOWN
            outcome.overall_confidence = 0.0

        outcome.reasoning = (
            f"Ensemble (pattern: {self.pattern_weight}, transformer: {self.transformer_weight}) | "
            f"Primary: {outcome.primary_label.value} (conf: {outcome.overall_confidence:.2f})"
        )

        return outcome


# ============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# ============================================================================

def classify_trial_outcomes_batch(
    texts: List[str],
    use_transformer: bool = False,
    verbose: bool = True
) -> List[TrialOutcome]:
    """
    Batch classify trial outcomes.

    Args:
        texts: List of PubMed abstracts or trial descriptions
        use_transformer: Whether to use transformer models (slower but more accurate)
        verbose: Print progress

    Returns:
        List of TrialOutcome objects
    """
    if use_transformer and TRANSFORMERS_AVAILABLE:
        classifier = EnsembleOutcomeClassifier(use_transformer=True)
    else:
        classifier = EnsembleOutcomeClassifier(use_transformer=False)

    outcomes = []
    for i, text in enumerate(texts):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(texts)} texts...")

        outcome = classifier.classify(text)
        outcomes.append(outcome)

    return outcomes


def augment_dataframe_with_outcomes(
    df: pd.DataFrame,
    text_column: str = "pubmed_text",
    use_transformer: bool = False
) -> pd.DataFrame:
    """
    Add sophisticated outcome classification to existing dataframe.

    This replaces the simple regex labeling in immport_cytometry_pipeline_full.py
    """
    df = df.copy()

    # Extract texts
    texts = df[text_column].fillna("").astype(str).tolist()

    print(f"Classifying {len(texts)} trial outcomes...")
    outcomes = classify_trial_outcomes_batch(texts, use_transformer=use_transformer)

    # Add columns
    df["outcome_labels_multi"] = [";".join([l.value for l in o.labels]) for o in outcomes]
    df["outcome_primary_label"] = [o.primary_label.value if o.primary_label else "unknown" for o in outcomes]
    df["outcome_confidence"] = [o.overall_confidence for o in outcomes]
    df["outcome_reasoning"] = [o.reasoning for o in outcomes]

    # Add individual label indicators (for filtering)
    for label in OutcomeLabel:
        df[f"has_{label.value}"] = [
            label in o.labels for o in outcomes
        ]

    # Add simplified categories for backward compatibility
    df["outcome_simple"] = df["outcome_primary_label"].apply(_simplify_label)

    return df


def _simplify_label(primary_label: str) -> str:
    """Convert detailed label to simple met/not_met/unknown for backward compatibility."""
    if primary_label in ["primary_endpoint_met", "superiority_shown", 
                         "clinical_benefit_demonstrated", "fda_approval_obtained"]:
        return "met"
    elif primary_label in ["primary_endpoint_not_met", "stopped_for_futility",
                           "no_clinical_benefit", "terminated_early"]:
        return "not_met"
    else:
        return "unknown"


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Demo/test of the classifier."""

    # Test cases
    test_texts = [
        """
        The trial met its primary endpoint of overall survival. 
        Patients treated with the experimental drug showed a statistically 
        significant improvement (p<0.001) compared to placebo, with a median 
        OS of 24.5 months vs 18.2 months (HR=0.72, 95% CI: 0.61-0.85). 
        The treatment was well-tolerated with manageable adverse events.
        """,

        """
        The study was terminated early for futility following a planned interim 
        analysis. The primary endpoint of progression-free survival was not met 
        (p=0.42), with no significant difference between treatment arms. 
        The data monitoring committee recommended discontinuation as the 
        conditional power was below 20%.
        """,

        """
        Although the primary endpoint was not achieved, the trial demonstrated 
        significant improvements in several secondary endpoints including quality 
        of life (p=0.003) and symptom burden (p=0.012). However, there were 
        serious safety concerns with 15% of patients experiencing grade 3-4 
        adverse events in the treatment arm vs 3% in placebo.
        """,
    ]

    print("="*80)
    print("TRIAL OUTCOME CLASSIFIER - DEMO")
    print("="*80)

    # Use pattern-based only (faster, no dependencies)
    classifier = EnsembleOutcomeClassifier(use_transformer=False)

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Text: {text[:150]}...")

        outcome = classifier.classify(text)

        print(f"\nPrimary Label: {outcome.primary_label.value}")
        print(f"Confidence: {outcome.overall_confidence:.2f}")
        print(f"All Labels: {[l.value for l in outcome.labels]}")
        print(f"Reasoning: {outcome.reasoning}")

        if outcome.evidence_spans:
            print(f"\nEvidence:")
            for label, spans in list(outcome.evidence_spans.items())[:3]:
                print(f"  {label.value}: {spans[0] if spans else 'N/A'}")


if __name__ == "__main__":
    main()

