from __future__ import annotations

import json
import os
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
SESSION.headers.update({"User-Agent": "immport-cytometry-pipeline-full/2.0"})

NCT_RE = re.compile(r"\bNCT\d{8}\b")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        if total_val is not None and len(out) >= int(total_val):
            break

        from_record += page_size

    # de-dup by study_accession
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
# CT.gov signals (Option A)
# -----------------------------
def ctgov_get_study(nct: str) -> Dict[str, Any]:
    return get_json(f"{CTGOV_BASE}/studies/{nct}")


def ctgov_status_whystopped(study_json: Dict[str, Any]) -> Tuple[str, str]:
    proto = (study_json or {}).get("protocolSection", {}) or {}
    status_mod = proto.get("statusModule", {}) or {}
    status = status_mod.get("overallStatus") or ""
    why = status_mod.get("whyStopped") or ""
    return str(status), str(why)


def ctgov_failure_score(status: str, why: str) -> Tuple[int, str]:
    """
    Option A scoring:
    - Terminated/Suspended/Withdrawn => +2
    - whyStopped contains futility/lack-of-efficacy => +4
    """
    status_l = (status or "").upper()
    why_l = (why or "").lower()
    score = 0
    reasons = []

    if status_l in ("TERMINATED", "SUSPENDED", "WITHDRAWN"):
        score += 2
        reasons.append(f"status={status_l}")

    if any(k in why_l for k in ["futility", "lack of efficacy", "inefficacy", "ineffective", "insufficient efficacy"]):
        score += 4
        reasons.append(f"whyStopped={why[:160]}")

    return score, " | ".join(reasons)


# -----------------------------
# PubMed signals (Option B fallback)
# -----------------------------
def efetch_pubmed_title_abstract(pubmed_ids: List[str], batch_size: int = 200, sleep: float = 0.34) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i in range(0, len(pubmed_ids), batch_size):
        batch = pubmed_ids[i : i + batch_size]
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
    (2, r"discontinued.*futility"),
    (2, r"terminated.*futility"),
]

SUCCESS_PATTERNS = [
    (4, r"met (the )?primary endpoint"),
    (4, r"primary endpoint was met"),
    (3, r"significantly (improved|reduced|increased)"),
    (3, r"demonstrated (a )?significant (benefit|improvement)"),
    (2, r"superior to (placebo|control)"),
    (2, r"resulted in significant"),
]


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
# Fusion
# -----------------------------
def fuse_labels(pub_label: str, pub_score: int, ct_score: int) -> Tuple[str, str]:
    """
    Final label + confidence rationale.
    Rules:
    - If PubMed clearly says met/not_met, trust that.
    - Else if CT.gov has strong futility/inefficacy signal => not_met
    - Else if CT.gov terminated/suspended => not_met_candidate
    - Else unknown
    """
    if pub_label == "not_met" and pub_score >= 4:
        return "not_met", "high: PubMed text"
    if pub_label == "met" and pub_score <= -4:
        return "met", "high: PubMed text"

    if ct_score >= 6:
        return "not_met", "med-high: CT.gov futility/inefficacy"
    if ct_score >= 2:
        return "not_met_candidate", "medium: CT.gov terminated/suspended (reason unclear)"

    return "unknown", "low: no deterministic signal"


# -----------------------------
# Sorting helpers (your requested order)
# -----------------------------
def tech_sort_key(assay_method: str) -> int:
    """
    Flow first, then CyTOF/Mass.
    """
    s = (assay_method or "").lower()
    if "flow" in s:
        return 0
    if "cytof" in s or "mass" in s:
        return 1
    return 9


def safe_int_year(x: Any) -> int:
    try:
        if pd.isna(x):
            return -1
        return int(str(x).strip())
    except Exception:
        return -1


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Discover assay methods
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

    working_cytof = [k for k, v in cy_counts.items() if v and v > 0]
    if not working_cytof:
        print("WARNING: No CyTOF methods returned hits. Proceeding with Flow Cytometry only.")

    search_plan = [(m, "Flow Cytometry") for m in FLOW_METHODS] + [(m, "CyTOF/Mass Cytometry") for m in working_cytof]

    # 2) Pull ImmPort studies + UI enrichment
    all_rows: List[Dict[str, Any]] = []
    for assay_method_query, assay_label in search_plan:
        studies = search_immport_studies(assay_method_query, page_size=200)
        print(f"{assay_label} via assayMethod={assay_method_query!r}: found {len(studies)} studies")

        for s in tqdm(studies, desc=f"Enrich {assay_label} ({assay_method_query})"):
            sdy = s.get("study_accession")
            if not sdy:
                continue

            try:
                ui = immport_ui_bundle(sdy)
                ncts = extract_nct_ids(ui)
                year, date_raw = extract_study_year_and_date_from_summary(ui.get("summary"))
                ui_err = ""
                has_studyfile = True
            except Exception as e:
                ncts = []
                year, date_raw = (None, None)
                ui_err = str(e)
                has_studyfile = False

            all_rows.append(
                {
                    "study_accession": sdy,
                    "technology": assay_label,                 # normalized label
                    "assay_method_raw_query": assay_method_query,  # exact term that matched ImmPort
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

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["study_accession", "technology"]).reset_index(drop=True)

    # 3) Fetch PubMed abstracts in batch
    df["pubmed_id"] = df["pubmed_id"].fillna("").astype(str)
    unique_pmids = sorted(
        set(
            pmid.strip()
            for cell in df["pubmed_id"].tolist()
            for pmid in cell.split(";")
            if pmid.strip().isdigit()
        )
    )
    print(f"Unique PubMed IDs: {len(unique_pmids)}")
    pub_texts = efetch_pubmed_title_abstract(unique_pmids, batch_size=200)

    # 4) Score PubMed per row
    pub_labels, pub_scores, pub_evidence = [], [], []
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
        pub_labels.append(lab)
        pub_scores.append(sc)
        pub_evidence.append(f"pmid={used} | {ev}")

    df["pubmed_label_scored"] = pub_labels
    df["pubmed_score"] = pub_scores
    df["pubmed_evidence_patterns"] = pub_evidence

    # 5) CT.gov scoring per row (best-of up to 5 NCTs)
    ct_statuses, ct_whys, ct_scores, ct_reasons = [], [], [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="CT.gov status/why"):
        ncts = [x for x in str(r["nct_ids"]).split(";") if x.startswith("NCT")]
        if not ncts:
            ct_statuses.append("")
            ct_whys.append("")
            ct_scores.append(0)
            ct_reasons.append("")
            continue

        best_score = 0
        best_status = ""
        best_why = ""
        best_reason = ""

        for nct in ncts[:5]:
            try:
                cj = ctgov_get_study(nct)
                status, why = ctgov_status_whystopped(cj)
                sc, rsn = ctgov_failure_score(status, why)
                if sc > best_score:
                    best_score = sc
                    best_status = status
                    best_why = why
                    best_reason = f"{nct} | {rsn}"
            except Exception:
                continue

        ct_statuses.append(best_status)
        ct_whys.append(best_why)
        ct_scores.append(best_score)
        ct_reasons.append(best_reason)

    df["ctgov_best_status"] = ct_statuses
    df["ctgov_best_whyStopped"] = ct_whys
    df["ctgov_failure_score"] = ct_scores
    df["ctgov_failure_reason"] = ct_reasons

    # 6) Fuse
    final_label, confidence = [], []
    for _, r in df.iterrows():
        lab, conf = fuse_labels(
            pub_label=str(r["pubmed_label_scored"]),
            pub_score=int(r["pubmed_score"]),
            ct_score=int(r["ctgov_failure_score"]),
        )
        final_label.append(lab)
        confidence.append(conf)

    df["final_outcome_label"] = final_label
    df["final_confidence"] = confidence

    # 7) Sort output by: year desc, technology order (Flow first), then label rank
    df["year_int"] = df["study_year"].apply(safe_int_year)
    df["tech_order"] = df["technology"].apply(tech_sort_key)

    label_rank = {"not_met": 0, "not_met_candidate": 1, "unknown": 2, "met": 3}
    df["label_rank"] = df["final_outcome_label"].map(label_rank).fillna(99).astype(int)

    df_sorted = df.sort_values(
        by=["year_int", "tech_order", "label_rank", "study_accession"],
        ascending=[False, True, True, True],
    ).drop(columns=["year_int", "tech_order", "label_rank"]).reset_index(drop=True)

    # 8) Write outputs
    out_full = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates_full.csv")
    out_sorted = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates_full_sorted.csv")

    df.to_csv(out_full, index=False)
    df_sorted.to_csv(out_sorted, index=False)

    print(f"Wrote: {out_full}")
    print(f"Wrote: {out_sorted}")
    print("\nFinal label counts:")
    print(df_sorted["final_outcome_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
