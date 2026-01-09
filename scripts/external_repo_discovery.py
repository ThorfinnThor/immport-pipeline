from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# -----------------------------
# Output
# -----------------------------
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUTPUT_DIR, "external_cytometry_candidates.csv")

# Optional: if ImmPort output exists, we match external NCTs to ImmPort trials
IMMPORT_RANKED = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates_full_ranked.csv")

# -----------------------------
# Controls
# -----------------------------
ZENODO_MAX_PAGES = int(os.environ.get("ZENODO_MAX_PAGES", "5"))
FIGSHARE_MAX_PAGES = int(os.environ.get("FIGSHARE_MAX_PAGES", "5"))
FIGSHARE_MAX_DETAILS = int(os.environ.get("FIGSHARE_MAX_DETAILS", "60"))

# Zenodo page size limits: anonymous max 25, authenticated max 100. 
ZENODO_PAGE_SIZE = int(os.environ.get("ZENODO_PAGE_SIZE", "25"))
FIGSHARE_PAGE_SIZE = int(os.environ.get("FIGSHARE_PAGE_SIZE", "50"))

ZENODO_TOKEN = os.environ.get("ZENODO_TOKEN", "").strip()
FIGSHARE_TOKEN = os.environ.get("FIGSHARE_TOKEN", "").strip()

# Keep Zenodo broad but we'll filter hard by cytometry signals
ZENODO_QUERY = os.environ.get(
    "ZENODO_QUERY",
    '( "flow cytometry" OR CyTOF OR "mass cytometry" OR FACS OR fcs OR "time-of-flight cytometry" )',
)

# Figshare search syntax: spaces behave like OR, and field queries exist. 
# We intentionally avoid parentheses/OR keywords and use multiple fallback queries.
FIGSHARE_QUERIES = [
    os.environ.get("FIGSHARE_QUERY", "").strip(),
    '"flow cytometry" fcs facs',
    '"flow cytometry" cytometry',
    'cytof "mass cytometry"',
    '"time-of-flight" cytometry',
]
FIGSHARE_QUERIES = [q for q in FIGSHARE_QUERIES if q]  # drop empty

# -----------------------------
# Cytometry detection (stricter to avoid false positives like badminton)
# -----------------------------
HIGH_SPECIFIC_TERMS = [
    "flow cytometry",
    "cytof",
    "mass cytometry",
    "facs",
    "fluorescence-activated cell sorting",
    "fcs",  # include but only as term (not as substring alone)
]

LOW_SPECIFIC_TERMS = [
    "time-of-flight cytometry",
    "cytometry by time-of-flight",
    "cytometry",  # lowest specificity
]

CYTO_EXTS_STRONG = {"fcs"}           # strong evidence
CYTO_EXTS_WEAK = {"wsp", "lmd"}      # weak evidence (workspaces etc.)

NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
PMID_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{6,9})", re.IGNORECASE)
PMID_TAG_RE = re.compile(r"\bPMID\s*[:#]?\s*(\d{6,9})\b", re.IGNORECASE)

# -----------------------------
# APIs
# -----------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "external-cytometry-discovery/2.0"})

CTGOV_BASE = "https://clinicaltrials.gov/api/v2"

# Figshare public API v2: https://api.figshare.com/v2; search: POST /v2/articles/search 
FIGSHARE_BASE = "https://api.figshare.com/v2"
ZENODO_BASE = "https://zenodo.org/api/records"


def backoff_sleep(attempt: int) -> None:
    time.sleep(min(12.0, 0.9 * (attempt + 1)))


def http_get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, retries: int = 4) -> Any:
    last_err = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                backoff_sleep(i)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            backoff_sleep(i)
    raise RuntimeError(f"GET failed: {url} params={params} err={last_err}")


def http_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, retries: int = 4) -> Any:
    last_err = None
    for i in range(retries):
        try:
            r = SESSION.post(url, json=payload, headers=headers, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                backoff_sleep(i)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            backoff_sleep(i)
    raise RuntimeError(f"POST failed: {url} payload_keys={list(payload.keys())} err={last_err}")


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").strip()


def extract_ncts(text: str) -> List[str]:
    if not text:
        return []
    return sorted(set(m.upper() for m in NCT_RE.findall(text)))


def extract_pmids(text: str) -> List[str]:
    if not text:
        return []
    pmids = set()
    for m in PMID_URL_RE.findall(text):
        pmids.add(m)
    for m in PMID_TAG_RE.findall(text):
        pmids.add(m)
    return sorted(pmids)


def file_ext(name: str) -> str:
    n = (name or "").strip()
    if "." not in n:
        return ""
    return n.rsplit(".", 1)[-1].lower()


def cytometry_signal(title: str, desc: str, keywords: str, file_names: List[str]) -> Tuple[float, str]:
    """
    Returns (confidence 0..1, reason).
    Stricter to reduce false positives.
    """
    t = (title or "").lower()
    d = (desc or "").lower()
    k = (keywords or "").lower()
    fn = " ".join(file_names).lower()

    reasons = []
    score = 0.0

    exts = {file_ext(x) for x in file_names if file_ext(x)}
    if exts & CYTO_EXTS_STRONG:
        score += 0.75
        reasons.append("has_fcs_file")

    if exts & CYTO_EXTS_WEAK:
        score += 0.15
        reasons.append("has_workspace_file")

    # High-specific terms
    high_hits = [term for term in HIGH_SPECIFIC_TERMS if term in t or term in d or term in k]
    if high_hits:
        score += 0.35
        reasons.append("high_terms:" + ",".join(sorted(set(high_hits))[:5]))

    # Low-specific terms (only help if something else already suggests cytometry)
    low_hits = [term for term in LOW_SPECIFIC_TERMS if term in t or term in d or term in k]
    if low_hits and score >= 0.35:
        score += 0.15
        reasons.append("low_terms:" + ",".join(sorted(set(low_hits))[:5]))

    # Filename terms (weak)
    if any(x in fn for x in ["fcs", "flowjo", "cytof"]):
        score += 0.10
        reasons.append("filename_terms")

    score = min(1.0, score)
    return score, " | ".join(reasons)


def tech_guess_from_text(text_blob: str, file_names: List[str]) -> str:
    tl = (text_blob or "").lower()
    if "cytof" in tl or "mass cytometry" in tl or "time-of-flight cytometry" in tl:
        return "CyTOF/Mass Cytometry"
    if "flow cytometry" in tl or "facs" in tl or any(file_ext(n) == "fcs" for n in file_names):
        return "Flow Cytometry"
    return "unknown"


# -----------------------------
# CT.gov verification for NCTs
# -----------------------------
_CT_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def ctgov_get_study(nct: str) -> Optional[Dict[str, Any]]:
    nct = nct.upper()
    if nct in _CT_CACHE:
        return _CT_CACHE[nct]

    try:
        data = http_get_json(f"{CTGOV_BASE}/studies/{nct}")
        _CT_CACHE[nct] = data
        return data
    except Exception:
        _CT_CACHE[nct] = None
        return None


def ctgov_overall_status(study_json: Dict[str, Any]) -> str:
    proto = (study_json or {}).get("protocolSection", {}) or {}
    status_mod = proto.get("statusModule", {}) or {}
    return str(status_mod.get("overallStatus") or "")


# -----------------------------
# ImmPort NCT matching
# -----------------------------
def load_immport_nct_map(path: str) -> Dict[str, List[str]]:
    """
    Returns map NCT -> [SDY...]
    """
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    if "nct_ids" not in df.columns or "study_accession" not in df.columns:
        return {}

    nct_map: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        sdy = str(r.get("study_accession", "")).strip()
        if not sdy:
            continue
        ncts = [x for x in str(r.get("nct_ids", "")).split(";") if x.startswith("NCT")]
        for n in ncts:
            nct_map.setdefault(n, []).append(sdy)
    return nct_map


# -----------------------------
# Zenodo
# -----------------------------
def zenodo_page_size() -> int:
    # enforce Zenodo limits
    if ZENODO_TOKEN:
        return min(100, max(1, ZENODO_PAGE_SIZE))
    return min(25, max(1, ZENODO_PAGE_SIZE))


def zenodo_search_records(query: str, max_pages: int) -> List[Dict[str, Any]]:
    size = zenodo_page_size()
    out: List[Dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "page": page,
            "size": size,
            "sort": "mostrecent",
        }
        if ZENODO_TOKEN:
            params["access_token"] = ZENODO_TOKEN

        data = http_get_json(ZENODO_BASE, params=params)
        hits = ((data or {}).get("hits", {}) or {}).get("hits", []) or []
        if not hits:
            break
        out.extend(hits)
        if len(hits) < size:
            break

    return out


def parse_zenodo_record(hit: Dict[str, Any]) -> Dict[str, Any]:
    rec_id = hit.get("id", "")
    links = hit.get("links", {}) or {}
    meta = hit.get("metadata", {}) or {}

    title = normalize_text(meta.get("title", ""))
    desc = normalize_text(meta.get("description", ""))
    keywords = meta.get("keywords", []) or []
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords_str = ";".join([normalize_text(k) for k in keywords if normalize_text(k)])

    pub_date = normalize_text(meta.get("publication_date", hit.get("created", "")))
    doi = normalize_text(meta.get("doi", ""))
    landing = normalize_text(links.get("html", ""))

    files = hit.get("files", []) or []
    file_names = [normalize_text(f.get("key", "")) for f in files if f.get("key")]
    file_exts = sorted({file_ext(n) for n in file_names if file_ext(n)})
    download_urls = []
    for f in files:
        l = (f.get("links", {}) or {}).get("self") or (f.get("links", {}) or {}).get("download")
        if l:
            download_urls.append(str(l))

    text_blob = " ".join([title, desc, keywords_str])
    ncts = extract_ncts(text_blob)
    pmids = extract_pmids(text_blob)

    conf, reason = cytometry_signal(title, desc, keywords_str, file_names)

    return {
        "source": "zenodo",
        "record_id": str(rec_id),
        "doi": doi,
        "title": title,
        "published_date": pub_date,
        "landing_page_url": landing,
        "description_snippet": desc[:500],
        "keywords": keywords_str,
        "file_names": ";".join(file_names[:50]),
        "file_extensions": ";".join(file_exts),
        "download_urls": " | ".join(download_urls[:20]),
        "nct_ids_extracted": ";".join(ncts),
        "pmid_extracted": ";".join(pmids),
        "technology_guess": tech_guess_from_text(text_blob, file_names),
        "cytometry_confidence": conf,
        "cytometry_reason": reason,
    }


# -----------------------------
# Figshare
# -----------------------------
def figshare_headers() -> Dict[str, str]:
    h: Dict[str, str] = {}
    if FIGSHARE_TOKEN:
        h["Authorization"] = f"token {FIGSHARE_TOKEN}"
    return h


def figshare_search_articles(search_for: str, max_pages: int, page_size: int) -> List[Dict[str, Any]]:
    """
    POST /v2/articles/search (public) 
    item_type=3 limits to datasets 
    """
    url = f"{FIGSHARE_BASE}/articles/search"
    headers = figshare_headers()
    out: List[Dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        payload = {
            "search_for": search_for,
            "item_type": 3,
            "page": page,
            "page_size": page_size,
            "order": "published_date",
            "order_direction": "desc",
        }
        try:
            data = http_post_json(url, payload=payload, headers=headers)
        except Exception as e:
            # Figshare returns 400 "Max page reached..." when too broad and user asks too many pages. 
            # We stop paging on any error (but we keep whatever we already got).
            print(f"[Figshare] Search error on page={page}: {e}")
            break

        if not isinstance(data, list) or not data:
            break

        out.extend(data)
        if len(data) < page_size:
            break

    return out


def figshare_get_article(article_id: int) -> Dict[str, Any]:
    return http_get_json(f"{FIGSHARE_BASE}/articles/{article_id}", headers=figshare_headers())


def parse_figshare_article(article_summary: Dict[str, Any]) -> Dict[str, Any]:
    aid = article_summary.get("id")
    if aid is None:
        return {}

    details = figshare_get_article(int(aid))

    title = normalize_text(details.get("title", article_summary.get("title", "")))
    desc = normalize_text(details.get("description", ""))
    doi = normalize_text(details.get("doi", article_summary.get("doi", "")))
    pub_date = normalize_text(details.get("published_date", article_summary.get("published_date", "")))
    landing = normalize_text(details.get("url_public_html", ""))

    tags = details.get("tags", []) or []
    if isinstance(tags, str):
        tags = [tags]
    tags_str = ";".join([normalize_text(t) for t in tags if normalize_text(t)])

    files = details.get("files", []) or []
    file_names = [normalize_text(f.get("name", "")) for f in files if f.get("name")]
    file_exts = sorted({file_ext(n) for n in file_names if file_ext(n)})
    download_urls = []
    for f in files:
        u = f.get("download_url")
        if u:
            download_urls.append(str(u))

    text_blob = " ".join([title, desc, tags_str])
    ncts = extract_ncts(text_blob)
    pmids = extract_pmids(text_blob)

    conf, reason = cytometry_signal(title, desc, tags_str, file_names)

    return {
        "source": "figshare",
        "record_id": str(aid),
        "doi": doi,
        "title": title,
        "published_date": pub_date,
        "landing_page_url": landing,
        "description_snippet": desc[:500],
        "keywords": tags_str,
        "file_names": ";".join(file_names[:50]),
        "file_extensions": ";".join(file_exts),
        "download_urls": " | ".join(download_urls[:20]),
        "nct_ids_extracted": ";".join(ncts),
        "pmid_extracted": ";".join(pmids),
        "technology_guess": tech_guess_from_text(text_blob, file_names),
        "cytometry_confidence": conf,
        "cytometry_reason": reason,
    }


# -----------------------------
# Enrichment: CT.gov + ImmPort matching
# -----------------------------
def enrich_trials(df: pd.DataFrame, immport_nct_map: Dict[str, List[str]]) -> pd.DataFrame:
    verified_any = []
    verified_ncts_col = []
    best_status_col = []
    matched_ncts_col = []
    matched_sdys_col = []

    for _, r in df.iterrows():
        ncts = [x for x in str(r.get("nct_ids_extracted", "")).split(";") if x.startswith("NCT")]

        # ImmPort match
        matched_ncts = []
        matched_sdys = []
        for n in ncts:
            if n in immport_nct_map:
                matched_ncts.append(n)
                matched_sdys.extend(immport_nct_map[n])

        matched_ncts = sorted(set(matched_ncts))
        matched_sdys = sorted(set(matched_sdys))

        # CT.gov verify (limit to 3 lookups per record for speed)
        verified = []
        best_status = ""
        for n in ncts[:3]:
            sj = ctgov_get_study(n)
            if sj:
                verified.append(n)
                if not best_status:
                    best_status = ctgov_overall_status(sj)

        verified = sorted(set(verified))
        verified_any.append(bool(verified))
        verified_ncts_col.append(";".join(verified))
        best_status_col.append(best_status)

        matched_ncts_col.append(";".join(matched_ncts))
        matched_sdys_col.append(";".join(matched_sdys))

    df["ctgov_any_verified"] = verified_any
    df["ctgov_verified_ncts"] = verified_ncts_col
    df["ctgov_best_status"] = best_status_col

    df["matched_immport_ncts"] = matched_ncts_col
    df["matched_immport_sdys"] = matched_sdys_col

    # Single indicator field for you
    def indicator(row: pd.Series) -> str:
        if bool(row.get("ctgov_any_verified")):
            return "ctgov_verified"
        if str(row.get("matched_immport_ncts", "")).strip():
            return "matched_immport"
        if str(row.get("nct_ids_extracted", "")).strip():
            return "nct_extracted_unverified"
        return ""

    df["clinical_trial_indicator"] = df.apply(indicator, axis=1)

    return df


def main() -> None:
    print("[Info] Loading ImmPort NCT map (if available)...")
    immport_nct_map = load_immport_nct_map(IMMPORT_RANKED)
    print(f"[Info] ImmPort NCTs loaded: {len(immport_nct_map)}")

    rows: List[Dict[str, Any]] = []

    # -------------------------
    # Zenodo
    # -------------------------
    print(f"\n[Zenodo] Query: {ZENODO_QUERY}")
    zen_hits = zenodo_search_records(ZENODO_QUERY, max_pages=ZENODO_MAX_PAGES)
    print(f"[Zenodo] Raw hits: {len(zen_hits)}")

    for h in zen_hits:
        row = parse_zenodo_record(h)
        if not row:
            continue

        # stricter filter to avoid junk
        if float(row.get("cytometry_confidence", 0.0)) >= 0.5:
            rows.append(row)

    print(f"[Zenodo] After cytometry filter: {sum(1 for r in rows if r['source']=='zenodo')}")

    # -------------------------
    # Figshare (fixed)
    # -------------------------
    fig_rows_before = len(rows)
    page_size = max(1, min(100, FIGSHARE_PAGE_SIZE))

    print("\n[Figshare] Trying queries in order until we get results...")
    fig_hits: List[Dict[str, Any]] = []
    for q in FIGSHARE_QUERIES:
        print(f"[Figshare] search_for={q!r}")
        hits = figshare_search_articles(q, max_pages=FIGSHARE_MAX_PAGES, page_size=page_size)
        print(f"[Figshare] results={len(hits)}")
        if hits:
            fig_hits = hits
            break

    if not fig_hits:
        print("[Figshare] No results found for any query (or rate-limited).")
    else:
        # Fetch details (bounded)
        for art in fig_hits[:FIGSHARE_MAX_DETAILS]:
            try:
                row = parse_figshare_article(art)
                if not row:
                    continue
                if float(row.get("cytometry_confidence", 0.0)) >= 0.55:
                    rows.append(row)
            except Exception:
                continue

    print(f"[Figshare] After cytometry filter: {len(rows) - fig_rows_before}")

    # -------------------------
    # Write output
    # -------------------------
    if not rows:
        pd.DataFrame([]).to_csv(OUT_CSV, index=False)
        print(f"\nWrote empty {OUT_CSV}")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["source", "record_id"]).reset_index(drop=True)

    # Enrich with CT.gov verification + ImmPort matching
    print("\n[Enrich] Verifying NCTs via CT.gov and matching to ImmPort...")
    df = enrich_trials(df, immport_nct_map)

    # Sort newest first
    df["published_date"] = df["published_date"].fillna("").astype(str)
    df = df.sort_values(by=["published_date", "source"], ascending=[False, True]).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")
    print("\nCounts by source:")
    print(df["source"].value_counts(dropna=False))
    print("\nClinical trial indicator counts:")
    print(df["clinical_trial_indicator"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
