from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# -----------------------------
# Endpoints
# -----------------------------
IMMPORT_STUDY_SEARCH = "https://www.immport.org/data/query/api/search/study"
IMMPORT_UI_BASE = "https://www.immport.org/data/query/ui"
CTGOV_BASE = "https://clinicaltrials.gov/api/v2"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "immport-cytometry-pipeline-full/1.0"})

NCT_RE = re.compile(r"\bNCT\d{8}\b")

# -----------------------------
# HTTP helpers
# -----------------------------
def get_json(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Any:
    last_err = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.8 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"GET JSON failed: {url} params={params} err={last_err}")


def get_text(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> str:
    last_err = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.8 * (i + 1))
                continue
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"GET TEXT failed: {url} params={params} err={last_err}")


# -----------------------------
# ImmPort helpers
# -----------------------------
def extract_nct_ids(obj: Any) -> List[str]:
    txt = json.dumps(obj, ensure_ascii=False)
    return sorted(set(NCT_RE.findall(txt)))


def _count_only(data: Dict[str, Any]) -> int:
    hits_block = data.get("hits", {}) or {}
    total = (hits_block.get("total", {}) or {}).get("value", 0)
    try:
        return int(total)
    except Exception:
        return 0


def probe_assay_methods(assay_methods: List[str]) -> Dict[str, int]:
    counts = {}
    for m in assay_methods:
        params = {
            "clinicalTrial": "Y",
            "assayMethod": m,
            "pageSize": 1,
            "fromRecord": 1,
            "sourceFields": "study_accession",
        }
        try:
            data = get_json(IMMPORT_STUDY_SEARCH, params=params)
            counts[m] = _count_only(data)
        except Exception:
            counts[m] = -1
    return counts


def search_immport_studies(
    assay_method: str,
    page_size: int = 200,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    from_record = 1
    out: List[Dict[str, Any]] = []

    while True:
        params = {
            "clinicalTrial": "Y",
            "assayMethod": assay_method,
            "pageSize": page_size,
            "fromRecord": from_record,
            "sourceFields": "study_accession,brief_title,pubmed_id,condition_or_disease,assay_method_count",
        }
        data = get_json(IMMPORT_STUDY_SEARCH, params=params)

        hits_block = data.get("hits", {}) or {}
        total_val = (hits_block.get("total", {}) or {}).get("value", None)

        hits = (hits_block.get("hits", []) or [])
        batch = []
        for h in hits:
            src = (h or {}).get("_source", {}) or {}
            if src:
                batch.append(src)

        if not batch:
            break

        out.extend(batch)

        if max_records and len(out) >= max_records:
            out = out[:max_records]
            break

        if total_val is not None and len(out) >= int(total_val):
            break

        from_record += page_size

    seen = set()
    dedup = []
    for x in out:
        sdy = x.get("study_accession")
        if not sdy or sdy in seen:
            continue
        seen.add(sdy)
        dedup.append(x)
    return dedup


def immport_ui_bundle(sdy: str) -> Dict[str, Any]:
    return {
        "summary": get_json(f"{IMMPORT_UI_BASE}/study/summary/{sdy}"),
        "design": get_json(f"{IMMPORT_UI_BASE}/study/design/{sdy}"),
        "studyfile": get_json(f"{IMMPORT_UI_BASE}/study/studyfile/{sdy}"),
    }


def _find_first_date_like(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    m = re.search(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", s)
    if m:
        return m.group(0).replace("/", "-")
    m = re.search(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])\b", s)
    if m:
        return m.group(0).replace("/", "-")
    return None


def extract_study_year_and_date_from_summary(summary_json: Any) -> Tuple[Optional[int], Optional[str]]:
    if summary_json is None:
        return (None, None)

    candidate_values = []
    if isinstance(summary_json, dict):
        for key in [
            "study_start_date", "studyStartDate", "start_date", "startDate", "actual_start_date",
            "study_end_date", "studyEndDate", "end_date", "endDate", "actual_end_date",
            "submission_date", "submissionDate", "release_date", "releaseDate"
        ]:
            if key in summary_json:
                candidate_values.append(summary_json.get(key))

        for key in ["study", "studySummary", "summary"]:
            if key in summary_json and isinstance(summary_json[key], dict):
                for k2 in [
                    "study_start_date", "studyStartDate", "start_date", "startDate", "actual_start_date",
                    "study_end_date", "studyEndDate", "end_date", "endDate", "actual_end_date",
                    "submission_date", "submissionDate", "release_date", "releaseDate"
                ]:
                    if k2 in summary_json[key]:
                        candidate_values.append(summary_json[key].get(k2))

    for v in candidate_values:
        d = _find_first_date_like(v)
        if d:
            return (int(d[:4]), d)

    txt = json.dumps(summary_json, ensure_ascii=False)
    d = _find_first_date_like(txt)
    if d:
        return (int(d[:4]), d)

    y = re.search(r"\b(19|20)\d{2}\b", txt)
    if y:
        return (int(y.group(0)), None)

    return (None, None)


# -----------------------------
# CT.gov (Option A)
# -----------------------------
FAIL_WHY_CUES = [
    "futility",
    "lack of efficacy",
    "inefficacy",
    "ineffective",
    "insufficient efficacy",
    "no efficacy",
]

FAIL_STATUS_CUES = {"TERMINATED", "SUSPENDED", "WITHDRAWN"}


def ctgov_get_study(nct: str) -> Dict[str, Any]:
    return get_json(f"{CTGOV_BASE}/studies/{nct}")


def ctgov_status_and_why(study_json: Dict[str, Any]) -> Tuple[str, str]:
    proto = (study_json.get("protocolSection") or {})
    status_mod = (proto.get("statusModule") or {})
    return (str(status_mod.get("overallStatus") or ""), str(status_mod.get("whyStopped") or ""))


def ctgov_option_a_label(nct: str) -> Tuple[str, str]:
    """
    Option A: Failure detection from CT.gov termination + whyStopped cues.
    Returns (label, evidence).
    label in: not_met, not_met_candidate, unknown
    """
    try:
        sj = ctgov_get_study(nct)
    except Exception as e:
        return ("unknown", f"ctgov_error:{e}")

    status, why = ctgov_status_and_why(sj)
    status_u = (status or "").upper()
    why_l = (why or "").lower()

    # strong failure: terminated-ish + futility/efficacy
    if any(c in why_l for c in FAIL_WHY_CUES):
        return ("not_met", f"{nct} | status={status_u} | whyStopped={why[:160]}")

    # candidate failure: terminated-ish but no reason
    if status_u in FAIL_STATUS_CUES:
        return ("not_met_candidate", f"{nct} | status={status_u} | whyStopped(empty/unclear)")

    return ("unknown", f"{nct} | status={status_u}")


# -----------------------------
# PubMed (Option B fallback)
# -----------------------------
FAIL_PATTERNS = [
    (4, r"did not meet (the )?primary endpoint"),
    (4, r"failed to meet (the )?primary endpoint"),
    (4, r"primary endpoint was not met"),
    (3, r"no significant difference"),
    (3, r"not significantly different"),
    (3, r"showed no (significant )?benefit"),
    (3, r"did not significantly (improve|reduce|increase)"),
    (3, r"failed to (improve|reduce|increase)"),
    (2, r"did not improve"),
    (2, r"no improvement in"),
    (2, r"no evidence of (benefit|efficacy)"),
]
SUCCESS_PATTERNS = [
    (4, r"met (the )?primary endpoint"),
    (4, r"primary endpoint was met"),
    (3, r"significantly (improved|reduced|increased)"),
    (3, r"demonstrated (a )?significant (benefit|improvement)"),
    (2, r"superior to (placebo|control)"),
    (2, r"resulted in significant"),
]


def pubmed_fetch_title_abstract(pmids: List[str], batch_size: int = 200, sleep: float = 0.34) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        xml_text = get_text(PUBMED_EFETCH, params=params)

        root = ET.fromstring(xml_text)
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//MedlineCitation/PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
            if not pmid:
                continue

            title = ""
            title_el = art.find(".//Article/ArticleTitle")
            if title_el is not None:
                title = "".join(title_el.itertext()).strip()

            abstract_parts = []
            for ab in art.findall(".//Article/Abstract/AbstractText"):
                abstract_parts.append("".join(ab.itertext()).strip())

            out[pmid] = " ".join([title] + abstract_parts).strip()

        time.sleep(sleep)
    return out


def pubmed_scored_label(text: str) -> Tuple[str, int, str]:
    if not text:
        return ("unknown", 0, "")

    t = " ".join(text.split())
    tl = t.lower()

    score = 0
    evidence = []

    for w, pat in FAIL_PATTERNS:
        if re.search(pat, tl):
            score += w
            evidence.append(f"+{w}:{pat}")
    for w, pat in SUCCESS_PATTERNS:
        if re.search(pat, tl):
            score -= w
            evidence.append(f"-{w}:{pat}")

    if score >= 4:
        return ("not_met", score, " | ".join(evidence)[:500])
    if score <= -4:
        return ("met", score, " | ".join(evidence)[:500])
    return ("unknown", score, " | ".join(evidence)[:500])


# -----------------------------
# Final fusion: Option A first, Option B fallback
# -----------------------------
def fuse_labels(ct_label: str, ct_evidence: str, pm_label: str, pm_score: int, pm_evidence: str) -> Tuple[str, str]:
    """
    Final label logic:
    1) If CT.gov says not_met (futility/lack of efficacy) => not_met
    2) Else if CT.gov says not_met_candidate (terminated etc.) => not_met_candidate
    3) Else fallback to PubMed:
         - pm_label not_met/met => use it
         - else unknown
    """
    if ct_label == "not_met":
        return ("not_met", f"CT.gov: {ct_evidence}")
    if ct_label == "not_met_candidate":
        # Keep candidate, but allow PubMed to override to met/not_met if it is strong
        if pm_label in ("not_met", "met"):
            return (pm_label, f"PubMed override (score={pm_score}): {pm_evidence}")
        return ("not_met_candidate", f"CT.gov: {ct_evidence}")
    # ct unknown
    if pm_label in ("not_met", "met"):
        return (pm_label, f"PubMed (score={pm_score}): {pm_evidence}")
    return ("unknown", "No deterministic CT.gov or PubMed signal")


# -----------------------------
# Main
# -----------------------------
def main():
    # Robust CyTOF discovery: probe multiple vocab strings and use the ones that return hits.
    FLOW_METHODS = ["Flow Cytometry"]
    CYTOF_METHOD_CANDIDATES = [
        "Mass Cytometry",
        "Mass cytometry",
        "CyTOF",
        "CYTOF",
        "Cytometry by time-of-flight",
        "Cytometry by Time-of-Flight",
        "Time-of-flight cytometry",
        "Mass cytometry (CyTOF)",
        "CyTOF (Mass Cytometry)",
    ]

    print("Probing CyTOF/Mass cytometry assayMethod vocabulary…")
    cy_counts = probe_assay_methods(CYTOF_METHOD_CANDIDATES)
    for k, v in cy_counts.items():
        print(f"  {k!r}: {v}")
    WORKING_CYTOF_METHODS = [k for k, v in cy_counts.items() if v and v > 0]
    if WORKING_CYTOF_METHODS:
        print(f"Using CyTOF methods: {WORKING_CYTOF_METHODS}")
    else:
        print("WARNING: No CyTOF methods returned hits. Proceeding with Flow Cytometry only.")

    all_rows: List[Dict[str, Any]] = []

    search_plan = [(m, "Flow Cytometry") for m in FLOW_METHODS] + [(m, "CyTOF/Mass Cytometry") for m in WORKING_CYTOF_METHODS]

    for assay_method, assay_label in search_plan:
        studies = search_immport_studies(assay_method, page_size=200)
        print(f"{assay_label} via assayMethod={assay_method!r}: found {len(studies)} studies")

        for s in tqdm(studies, desc=f"Enrich {assay_label} ({assay_method})"):
            sdy = s.get("study_accession")
            if not sdy:
                continue

            # UI bundle
            try:
                ui = immport_ui_bundle(sdy)
                ncts = extract_nct_ids(ui)
                year, date_raw = extract_study_year_and_date_from_summary(ui.get("summary"))
                ui_err = ""
                has_studyfile = True
            except Exception as e:
                ui = {"error": str(e)}
                ncts = []
                year, date_raw = (None, None)
                ui_err = str(e)
                has_studyfile = False

            all_rows.append(
                {
                    "study_accession": sdy,
                    "assay_method": assay_label,
                    "assay_method_raw_query": assay_method,
                    "brief_title": s.get("brief_title", ""),
                    "pubmed_id": ";".join(s.get("pubmed_id", [])) if isinstance(s.get("pubmed_id"), list) else (s.get("pubmed_id") or ""),
                    "condition_or_disease": ";".join(s.get("condition_or_disease", [])) if isinstance(s.get("condition_or_disease"), list) else (s.get("condition_or_disease") or ""),
                    "nct_ids": ";".join(ncts),
                    "study_year": year if year is not None else "",
                    "study_date_raw": date_raw if date_raw is not None else "",
                    "ui_bundle_has_studyfile": has_studyfile,
                    "ui_bundle_error": ui_err,
                }
            )

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["study_accession", "assay_method"]).reset_index(drop=True)

    # -----------------------------
    # Option A: CT.gov status+whyStopped failure detection
    # -----------------------------
    ct_labels = []
    ct_evidences = []
    ct_nct_used = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="CT.gov Option A labeling"):
        ncts = [x for x in str(r["nct_ids"]).split(";") if x.startswith("NCT")]
        if not ncts:
            ct_labels.append("unknown")
            ct_evidences.append("no NCT id")
            ct_nct_used.append("")
            continue

        best_label = "unknown"
        best_ev = ""
        best_nct = ""

        # Choose strongest signal across up to 5 NCTs
        # Priority: not_met > not_met_candidate > unknown
        priority = {"not_met": 2, "not_met_candidate": 1, "unknown": 0}

        for nct in ncts[:5]:
            lab, ev = ctgov_option_a_label(nct)
            if priority.get(lab, 0) > priority.get(best_label, 0):
                best_label, best_ev, best_nct = lab, ev, nct
            if best_label == "not_met":
                break

        ct_labels.append(best_label)
        ct_evidences.append(best_ev)
        ct_nct_used.append(best_nct)

    df["ctgov_option_a_label"] = ct_labels
    df["ctgov_option_a_evidence"] = ct_evidences
    df["ctgov_nct_used"] = ct_nct_used

    # -----------------------------
    # Option B fallback: PubMed scored label
    # -----------------------------
    # Collect unique PMIDs
    df["pubmed_id"] = df["pubmed_id"].fillna("").astype(str)
    unique_pmids = sorted(
        set(
            pmid.strip()
            for cell in df["pubmed_id"].tolist()
            for pmid in cell.split(";")
            if pmid.strip().isdigit()
        )
    )

    print(f"Unique PubMed IDs to fetch: {len(unique_pmids)}")
    pub_texts = pubmed_fetch_title_abstract(unique_pmids, batch_size=200)

    pm_labels, pm_scores, pm_evidences = [], [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="PubMed scoring"):
        pmids = [p for p in str(r["pubmed_id"]).split(";") if p.strip().isdigit()]
        text = ""
        used = ""
        for p in pmids:
            if pub_texts.get(p):
                text = pub_texts[p]
                used = p
                break
        lab, sc, ev = pubmed_scored_label(text)
        pm_labels.append(lab)
        pm_scores.append(sc)
        pm_evidences.append(f"pmid={used} | {ev}")

    df["pubmed_label_scored"] = pm_labels
    df["pubmed_score"] = pm_scores
    df["pubmed_evidence"] = pm_evidences

    # -----------------------------
    # Final label: Option A first, Option B fallback/override
    # -----------------------------
    final_labels = []
    final_evidence = []

    for _, r in df.iterrows():
        lab, ev = fuse_labels(
            ct_label=str(r["ctgov_option_a_label"]),
            ct_evidence=str(r["ctgov_option_a_evidence"]),
            pm_label=str(r["pubmed_label_scored"]),
            pm_score=int(r["pubmed_score"]) if str(r["pubmed_score"]).strip() != "" else 0,
            pm_evidence=str(r["pubmed_evidence"]),
        )
        final_labels.append(lab)
        final_evidence.append(ev)

    df["final_outcome_label"] = final_labels
    df["final_outcome_evidence"] = final_evidence

    # Rank for convenience
    rank_map = {"not_met": 0, "not_met_candidate": 1, "unknown": 2, "met": 3}
    df["rank"] = df["final_outcome_label"].map(rank_map).fillna(99).astype(int)
    df_ranked = df.sort_values(["rank", "study_year"], ascending=[True, False]).reset_index(drop=True)

    # Write outputs
    df.to_csv("immport_cytometry_candidates_full.csv", index=False)
    df_ranked.to_csv("immport_cytometry_candidates_full_ranked.csv", index=False)

    print("Wrote immport_cytometry_candidates_full.csv")
    print("Wrote immport_cytometry_candidates_full_ranked.csv")
    print("\nFinal label counts:")
    print(df["final_outcome_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
