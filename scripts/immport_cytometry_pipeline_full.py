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
# CONFIG (env vars)
# -----------------------------
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

# Keep <= 100 rows (your request)
MAX_ROWS = int(os.environ.get("MAX_ROWS", "100"))

# If 1: require ImmPort clinicalTrial=Y (your original behavior)
CLINICAL_TRIAL_ONLY = os.environ.get("CLINICAL_TRIAL_ONLY", "1") in ("1", "true", "True", "yes", "YES")

# -----------------------------
# Endpoints
# -----------------------------
IMMPORT_STUDY_SEARCH = "https://www.immport.org/data/query/api/search/study"
IMMPORT_UI_BASE = "https://www.immport.org/data/query/ui"
CTGOV_BASE = "https://clinicaltrials.gov/api/v2"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "immport-cytometry-pipeline-nogemini/1.0"})

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
# ImmPort search
# -----------------------------
def probe_assay_methods(assay_methods: List[str]) -> Dict[str, int]:
    counts = {}
    for m in assay_methods:
        params = {
            "pageSize": 1,
            "fromRecord": 1,
            "assayMethod": m,
            "sourceFields": "study_accession",
        }
        if CLINICAL_TRIAL_ONLY:
            params["clinicalTrial"] = "Y"
        try:
            data = get_json(IMMPORT_STUDY_SEARCH, params=params)
            total = (data.get("hits", {}).get("total", {}) or {}).get("value", 0)
            counts[m] = int(total) if total is not None else 0
        except Exception:
            counts[m] = -1
    return counts


def search_immport_studies(assay_method: str, page_size: int = 200) -> List[Dict[str, Any]]:
    from_record = 1
    out: List[Dict[str, Any]] = []

    while True:
        params: Dict[str, Any] = {
            "assayMethod": assay_method,
            "pageSize": page_size,
            "fromRecord": from_record,
            "sourceFields": "study_accession,brief_title,pubmed_id,condition_or_disease,assay_method_count,initial_data_release_date,latest_data_release_date",
        }
        if CLINICAL_TRIAL_ONLY:
            params["clinicalTrial"] = "Y"

        data = get_json(IMMPORT_STUDY_SEARCH, params=params)
        hits_block = data.get("hits", {}) or {}
        total_val = (hits_block.get("total", {}) or {}).get("value", None)
        hits = hits_block.get("hits", []) or []

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


def extract_nct_ids(obj: Any) -> List[str]:
    txt = json.dumps(obj, ensure_ascii=False)
    return sorted(set(NCT_RE.findall(txt)))


# -----------------------------
# PubMed
# -----------------------------
def efetch_pubmed_title_abstract(pubmed_ids: List[str], batch_size: int = 200, sleep: float = 0.34) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not pubmed_ids:
        return out

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
    (2, r"no evidence of (benefit|efficacy)"),
    (2, r"(discontinued|terminated).*(futility|lack of efficacy|inefficacy)"),
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

    tl = " ".join(text.split()).lower()
    score = 0
    evidence: List[str] = []

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
# ClinicalTrials.gov (Option A)
# -----------------------------
def ctgov_get_study(nct: str) -> Dict[str, Any]:
    return get_json(f"{CTGOV_BASE}/studies/{nct}")


def ctgov_status_whystopped(study_json: Dict[str, Any]) -> Tuple[str, str]:
    proto = (study_json or {}).get("protocolSection", {}) or {}
    status_mod = proto.get("statusModule", {}) or {}
    return str(status_mod.get("overallStatus") or ""), str(status_mod.get("whyStopped") or "")


def ctgov_failure_score(status: str, why: str) -> Tuple[int, str]:
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
# Fusion + sort/cap
# -----------------------------
def fuse_baseline(pub_label: str, pub_score: int, ct_score: int) -> Tuple[str, str]:
    if pub_label == "not_met" and pub_score >= 4:
        return "not_met", "high: PubMed text"
    if pub_label == "met" and pub_score <= -4:
        return "met", "high: PubMed text"

    if ct_score >= 6:
        return "not_met", "med-high: CT.gov futility/inefficacy"
    if ct_score >= 2:
        return "not_met_candidate", "medium: CT.gov terminated/suspended (reason unclear)"

    return "unknown", "low: no deterministic signal"


def to_date_sort_key(s: Any) -> Tuple[int, int, int]:
    """
    Sort key for initial_data_release_date (newest first later by reverse=True):
    Returns (year, month, day); missing => (0,0,0)
    Accepts 'YYYY-MM-DD' or similar.
    """
    if s is None:
        return (0, 0, 0)
    txt = str(s).strip()
    m = re.search(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", txt)
    if not m:
        return (0, 0, 0)
    parts = m.group(0).split("-")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def cap_latest(df_sorted: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df_sorted) <= max_rows:
        return df_sorted
    return df_sorted.head(max_rows).reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Keep the stable assay strings you already observed working
    FLOW_CANDIDATES = [
        "Flow Cytometry",
        "Flow cytometry",
        "FACS",
        "Fluorescence-activated cell sorting",
        "Fluorescence Activated Cell Sorting",
    ]
    CYTOF_CANDIDATES = [
        "CyTOF",
        "CYTOF",
        "Mass Cytometry",
        "Mass cytometry",
        "Cytometry by time-of-flight",
        "Time-of-flight cytometry",
        "Mass cytometry (CyTOF)",
        "CyTOF (Mass Cytometry)",
    ]

    print("Probing Flow assay methods…")
    flow_counts = probe_assay_methods(FLOW_CANDIDATES)
    for k, v in flow_counts.items():
        print(f"  {k!r}: {v}")
    working_flow = [k for k, v in flow_counts.items() if v and v > 0]
    if not working_flow:
        working_flow = ["Flow Cytometry"]

    print("\nProbing CyTOF assay methods…")
    cy_counts = probe_assay_methods(CYTOF_CANDIDATES)
    for k, v in cy_counts.items():
        print(f"  {k!r}: {v}")
    working_cytof = [k for k, v in cy_counts.items() if v and v > 0]
    if not working_cytof:
        print("WARNING: No CyTOF assay method variant returned hits. Proceeding without CyTOF.")

    search_plan: List[Tuple[str, str]] = []
    for m in working_flow:
        search_plan.append((m, "Flow Cytometry"))
    for m in working_cytof:
        search_plan.append((m, "CyTOF / Mass Cytometry"))

    all_rows: List[Dict[str, Any]] = []
    seen = set()  # (SDY, technology)

    for assay_method_query, technology in search_plan:
        studies = search_immport_studies(assay_method_query, page_size=200)
        print(f"\n{technology} via assayMethod={assay_method_query!r}: found {len(studies)} studies")

        for s in tqdm(studies, desc=f"Enrich {technology} ({assay_method_query})"):
            sdy = s.get("study_accession")
            if not sdy:
                continue

            key = (sdy, technology)
            if key in seen:
                continue
            seen.add(key)

            # NCT IDs from UI bundle
            try:
                ui = immport_ui_bundle(sdy)
                ncts = extract_nct_ids(ui)
                ui_err = ""
                has_studyfile = True
            except Exception as e:
                ncts = []
                ui_err = str(e)
                has_studyfile = False

            all_rows.append(
                {
                    "study_accession": sdy,
                    "technology": technology,
                    "assay_method_raw_query": assay_method_query,
                    "brief_title": s.get("brief_title", ""),
                    "pubmed_id": ";".join(s.get("pubmed_id", [])) if isinstance(s.get("pubmed_id"), list) else (s.get("pubmed_id") or ""),
                    "condition_or_disease": ";".join(s.get("condition_or_disease", [])) if isinstance(s.get("condition_or_disease"), list) else (s.get("condition_or_disease") or ""),
                    "nct_ids": ";".join(ncts),
                    "initial_data_release_date": s.get("initial_data_release_date", ""),
                    "latest_data_release_date": s.get("latest_data_release_date", ""),
                    "ui_bundle_has_studyfile": has_studyfile,
                    "ui_bundle_error": ui_err,
                }
            )

    df = pd.DataFrame(all_rows).reset_index(drop=True)

    # PubMed fetch + scoring
    df["pubmed_id"] = df["pubmed_id"].fillna("").astype(str)
    unique_pmids = sorted(
        set(
            pmid.strip()
            for cell in df["pubmed_id"].tolist()
            for pmid in cell.split(";")
            if pmid.strip().isdigit()
        )
    )
    print(f"\nUnique PubMed IDs: {len(unique_pmids)}")
    pub_texts = efetch_pubmed_title_abstract(unique_pmids, batch_size=200)

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
        pub_evidence.append(f"pmid={used} | {ev}".strip())

    df["pubmed_label_scored"] = pub_labels
    df["pubmed_score"] = pub_scores
    df["pubmed_evidence_patterns"] = pub_evidence

    # CT.gov scoring
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

    # Final baseline fusion
    final_labels, final_conf = [], []
    for _, r in df.iterrows():
        lab, conf = fuse_baseline(
            pub_label=str(r["pubmed_label_scored"]),
            pub_score=int(r["pubmed_score"]),
            ct_score=int(r["ctgov_failure_score"]),
        )
        final_labels.append(lab)
        final_conf.append(conf)

    df["final_outcome_label"] = final_labels
    df["final_confidence"] = final_conf

    # ORDER FINAL CSV BY initial_data_release_date (newest to oldest)
    df["initial_date_key"] = df["initial_data_release_date"].apply(to_date_sort_key)
    df = df.sort_values(by=["initial_date_key", "study_accession"], ascending=[False, True]).drop(columns=["initial_date_key"])

    # Cap to <= MAX_ROWS (newest first)
    df_ranked = cap_latest(df, MAX_ROWS)

    # Outputs
    out_full = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates_full.csv")
    out_ranked = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates_full_ranked.csv")

    df.to_csv(out_full, index=False)
    df_ranked.to_csv(out_ranked, index=False)

    print(f"\nWrote: {out_full}")
    print(f"Wrote: {out_ranked}")
    print("\nFinal label counts (ranked):")
    print(df_ranked["final_outcome_label"].value_counts(dropna=False))
    print("\nTechnology counts (ranked):")
    print(df_ranked["technology"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
