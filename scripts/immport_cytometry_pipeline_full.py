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

from pipeline_integration import integrate_into_pipeline

# -----------------------------
# CONFIG (env vars)
# -----------------------------

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ROWS = int(os.environ.get("MAX_ROWS", "100"))  # cap for the "final ranked" output
CLINICAL_TRIAL_ONLY = os.environ.get("CLINICAL_TRIAL_ONLY", "1") in (
    "1",
    "true",
    "True",
    "yes",
    "YES",
)

# -----------------------------
# Endpoints
# -----------------------------

IMMPORT_STUDY_SEARCH = "https://www.immport.org/data/query/api/search/study"
IMMPORT_UI_BASE = "https://www.immport.org/data/query/ui"
CTGOV_BASE = "https://clinicaltrials.gov/api/v2"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "immport-cytometry-pipeline-integrated/5.0"})

NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)

# -----------------------------
# HTTP helpers
# -----------------------------


def get_json(
    url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3
) -> Any:
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


def get_text(
    url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3
) -> str:
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
    return sorted(set(re.findall(r"\bNCT\d{8}\b", txt, flags=re.IGNORECASE)))


def _count_only(data: Dict[str, Any]) -> int:
    hits_block = data.get("hits", {}) or {}
    total = (hits_block.get("total", {}) or {}).get("value", 0)
    try:
        return int(total)
    except Exception:
        return 0


def probe_assay_methods(assay_methods: List[str]) -> Dict[str, int]:
    """
    Probe which ImmPort assayMethod strings return hits.
    """
    counts: Dict[str, int] = {}
    for m in assay_methods:
        params: Dict[str, Any] = {
            "assayMethod": m,
            "pageSize": 1,
            "fromRecord": 1,
            "sourceFields": "study_accession",
        }
        if CLINICAL_TRIAL_ONLY:
            params["clinicalTrial"] = "Y"
        try:
            data = get_json(IMMPORT_STUDY_SEARCH, params=params)
            counts[m] = _count_only(data)
        except Exception:
            counts[m] = -1
    return counts


def search_immport_studies(assay_method: str, page_size: int = 200) -> List[Dict[str, Any]]:
    """
    ImmPort /search/study pagination using fromRecord (1-indexed).
    We also request initial/latest_data_release_date so we can sort later.
    """
    from_record = 1
    out: List[Dict[str, Any]] = []

    while True:
        params: Dict[str, Any] = {
            "assayMethod": assay_method,
            "pageSize": page_size,
            "fromRecord": from_record,
            "sourceFields": ",".join(
                [
                    "study_accession",
                    "brief_title",
                    "pubmed_id",
                    "condition_or_disease",
                    "assay_method_count",
                    "initial_data_release_date",
                    "latest_data_release_date",
                ]
            ),
        }
        if CLINICAL_TRIAL_ONLY:
            params["clinicalTrial"] = "Y"

        data = get_json(IMMPORT_STUDY_SEARCH, params=params)
        hits_block = data.get("hits", {}) or {}
        total_val = (hits_block.get("total", {}) or {}).get("value", None)
        hits = (hits_block.get("hits", []) or [])

        batch: List[Dict[str, Any]] = []
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

    # de-dup by SDY
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
    """
    Return first ISO-like date string found in a JSON-serializable value.
    Accepts: YYYY-MM-DD, YYYY/MM/DD, or YYYY-MM.
    """
    if value is None:
        return None
    s = str(value)
    m = re.search(
        r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", s
    )
    if m:
        return m.group(0).replace("/", "-")
    m = re.search(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])\b", s)
    if m:
        return m.group(0).replace("/", "-")
    return None


def extract_study_year_and_date_from_summary(
    summary_json: Any,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Try common keys, then regex scan.
    Returns (year, raw_date_str).
    """
    if summary_json is None:
        return (None, None)

    candidate_values = []
    if isinstance(summary_json, dict):
        for key in [
            "study_start_date",
            "studyStartDate",
            "start_date",
            "startDate",
            "actual_start_date",
            "study_end_date",
            "studyEndDate",
            "end_date",
            "endDate",
            "actual_end_date",
            "submission_date",
            "submissionDate",
            "release_date",
            "releaseDate",
        ]:
            if key in summary_json:
                candidate_values.append(summary_json.get(key))

        for key in ["study", "studySummary", "summary"]:
            if key in summary_json and isinstance(summary_json[key], dict):
                for k2 in [
                    "study_start_date",
                    "studyStartDate",
                    "start_date",
                    "startDate",
                    "actual_start_date",
                    "study_end_date",
                    "studyEndDate",
                    "end_date",
                    "endDate",
                    "actual_end_date",
                    "submission_date",
                    "submissionDate",
                    "release_date",
                    "releaseDate",
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
# PubMed fetch (title+abstract)
# -----------------------------


def efetch_pubmed_title_abstract(
    pubmed_ids: List[str],
    batch_size: int = 200,
    sleep: float = 0.34,
) -> Dict[str, str]:
    """
    Fetch PubMed title + abstract in batches. Returns {pmid: combined_text}.
    """
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


# -----------------------------
# Sorting helper
# -----------------------------


def parse_iso_date_tuple(s: Any) -> Tuple[int, int, int]:
    """
    Returns (YYYY, MM, DD) for sorting. Missing/invalid -> (0,0,0).
    """
    if s is None:
        return (0, 0, 0)
    txt = str(s).strip()
    m = re.search(
        r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", txt
    )
    if not m:
        return (0, 0, 0)
    y, mo, d = m.group(0).split("-")
    return (int(y), int(mo), int(d))


# -----------------------------
# MAIN
# -----------------------------


def main() -> None:
    # 1) Determine working assayMethod strings (probe)
    FLOW_CANDIDATES = [
        "Flow Cytometry",
        "Flow cytometry",
        "FACS",
        "Fluorescence-activated cell sorting",
        "Fluorescence Activated Cell Sorting",
        "Fluorescence activated cell sorting (FACS)",
    ]

    CYTOF_CANDIDATES = [
        "CyTOF",
        "CYTOF",
        "Mass Cytometry",
        "Mass cytometry",
        "Cytometry by time-of-flight",
        "Cytometry by Time-of-Flight",
        "Time-of-flight cytometry",
        "Mass cytometry (CyTOF)",
        "CyTOF (Mass Cytometry)",
    ]

    print("Probing Flow Cytometry assayMethod strings...")
    flow_counts = probe_assay_methods(FLOW_CANDIDATES)
    for k, v in flow_counts.items():
        print(f" {k!r}: {v}")
    working_flow = [k for k, v in flow_counts.items() if v and v > 0]
    if not working_flow:
        working_flow = ["Flow Cytometry"]
        print("WARNING: No Flow synonym returned hits; falling back to 'Flow Cytometry'.")

    print("\nProbing CyTOF/Mass Cytometry assayMethod strings...")
    cy_counts = probe_assay_methods(CYTOF_CANDIDATES)
    for k, v in cy_counts.items():
        print(f" {k!r}: {v}")
    working_cytof = [k for k, v in cy_counts.items() if v and v > 0]
    if not working_cytof:
        print("WARNING: No CyTOF synonym returned hits; proceeding without CyTOF.")

    search_plan: List[Tuple[str, str]] = []
    for m in working_flow:
        search_plan.append((m, "Flow Cytometry"))
    for m in working_cytof:
        search_plan.append((m, "CyTOF/Mass Cytometry"))

    # 2) ImmPort search + UI enrichment
    all_rows: List[Dict[str, Any]] = []
    seen = set()

    for assay_method_query, technology in search_plan:
        studies = search_immport_studies(assay_method_query, page_size=200)
        print(
            f"\n{technology} via assayMethod={assay_method_query!r}: "
            f"found {len(studies)} studies"
        )
        for s in tqdm(studies, desc=f"Enrich {technology} ({assay_method_query})"):
            sdy = s.get("study_accession")
            if not sdy:
                continue
            key = (sdy, technology)
            if key in seen:
                continue
            seen.add(key)

            try:
                ui = immport_ui_bundle(sdy)
                ncts = extract_nct_ids(ui)
                year, date_raw = extract_study_year_and_date_from_summary(
                    ui.get("summary")
                )
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
                    "technology": technology,
                    "assay_method_raw_query": assay_method_query,
                    "brief_title": s.get("brief_title", ""),
                    "pubmed_id": ";".join(s.get("pubmed_id", []))
                    if isinstance(s.get("pubmed_id"), list)
                    else (s.get("pubmed_id") or ""),
                    "condition_or_disease": ";".join(
                        s.get("condition_or_disease", [])
                    )
                    if isinstance(s.get("condition_or_disease"), list)
                    else (s.get("condition_or_disease") or ""),
                    "nct_ids": ";".join(ncts),
                    "study_year": year if year is not None else "",
                    "study_date_raw": date_raw if date_raw is not None else "",
                    "initial_data_release_date": s.get("initial_data_release_date", ""),
                    "latest_data_release_date": s.get("latest_data_release_date", ""),
                    "ui_bundle_has_studyfile": has_studyfile,
                    "ui_bundle_error": ui_err,
                }
            )

    df = pd.DataFrame(all_rows).reset_index(drop=True)
    out_candidates = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates.csv")
    df.to_csv(out_candidates, index=False)
    print(f"\nWrote {out_candidates} ({len(df)} rows)")

    # 3) PubMed fetch once (unique PMIDs)
    df["pubmed_id"] = df["pubmed_id"].fillna("").astype(str)
    unique_pmids = sorted(
        set(
            pmid.strip()
            for cell in df["pubmed_id"].tolist()
            for pmid in str(cell).split(";")
            if pmid.strip().isdigit()
        )
    )
    print(f"Unique PubMed IDs: {len(unique_pmids)}")
    pub_texts = efetch_pubmed_title_abstract(unique_pmids, batch_size=200)

    # 4–6) Enhanced PubMed + CT.gov classification and fusion
    df = integrate_into_pipeline(df, pub_texts, OUTPUT_DIR)

    # 7) Outputs
    out_full = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates_full.csv")
    df.to_csv(out_full, index=False)
    print(f"Wrote {out_full}")

    # A) Final CSV ordered by initial_data_release_date newest -> oldest
    df_sorted = df.copy()
    df_sorted["initial_date_key"] = df_sorted[
        "initial_data_release_date"
    ].apply(parse_iso_date_tuple)
    df_sorted["latest_date_key"] = df_sorted[
        "latest_data_release_date"
    ].apply(parse_iso_date_tuple)
    df_sorted = (
        df_sorted.sort_values(
            by=["initial_date_key", "latest_date_key", "study_accession"],
            ascending=[False, False, True],
        )
        .drop(columns=["initial_date_key", "latest_date_key"])
        .reset_index(drop=True)
    )

    df_final = df_sorted.head(MAX_ROWS).copy()
    out_final_ranked = os.path.join(
        OUTPUT_DIR, "immport_cytometry_candidates_full_ranked.csv"
    )
    df_final.to_csv(out_final_ranked, index=False)
    print(
        f"Wrote {out_final_ranked} "
        f"(sorted by initial_data_release_date, capped to MAX_ROWS={MAX_ROWS})"
    )

    # B) Failed-ranked output (robust to missing ctgov_failure_score)
    df_failed = df.copy()
    rank_map = {
        "not_met": 0,
        "not_met_futility": 0,
        "not_met_early_term": 0,
        "terminated_safety": 0,
        "likely_not_met": 1,
        "not_met_candidate": 2,
        "unknown": 3,
        "met": 4,
        "likely_met": 4,
        "clinical_benefit": 5,
    }
    df_failed["rank"] = (
        df_failed["final_outcome_label"].map(rank_map).fillna(99).astype(int)
    )

    sort_cols = ["rank"]
    sort_asc = [True]

    # Use CT.gov score if available
    if "ctgov_failure_score" in df_failed.columns:
        sort_cols.append("ctgov_failure_score")
        sort_asc.append(False)

    # Always use pubmed_score as secondary
    sort_cols.append("pubmed_score")
    sort_asc.append(False)

    df_failed = df_failed.sort_values(
        sort_cols,
        ascending=sort_asc,
    ).reset_index(drop=True)

    out_failed = os.path.join(OUTPUT_DIR, "immport_failed_trials_ranked.csv")
    df_failed.to_csv(out_failed, index=False)
    print(f"Wrote {out_failed}")

    print("\nFinal label counts (full):")
    print(df["final_outcome_label"].value_counts(dropna=False))

    print("\nFinal label counts (failed-ranked):")
    print(df_failed["final_outcome_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
