from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many pages to pull per source (keep conservative for rate limits)
ZENODO_MAX_PAGES = int(os.environ.get("ZENODO_MAX_PAGES", "5"))
FIGSHARE_MAX_PAGES = int(os.environ.get("FIGSHARE_MAX_PAGES", "5"))

# Zenodo: anonymous page size is effectively limited (see Zenodo announcement).
# We enforce 25 unless you provide ZENODO_TOKEN (then allow up to 100).
ZENODO_PAGE_SIZE_DEFAULT = int(os.environ.get("ZENODO_PAGE_SIZE", "25"))
FIGSHARE_PAGE_SIZE_DEFAULT = int(os.environ.get("FIGSHARE_PAGE_SIZE", "50"))

# Optional tokens
ZENODO_TOKEN = os.environ.get("ZENODO_TOKEN", "").strip()
FIGSHARE_TOKEN = os.environ.get("FIGSHARE_TOKEN", "").strip()

# Queries (override if you want)
ZENODO_QUERY = os.environ.get(
    "ZENODO_QUERY",
    '( "flow cytometry" OR cytometry OR CyTOF OR "mass cytometry" OR FACS OR fcs OR "time-of-flight" )',
)
FIGSHARE_QUERY = os.environ.get(
    "FIGSHARE_QUERY",
    '("flow cytometry" OR cytometry OR CyTOF OR "mass cytometry" OR FACS OR fcs OR "time-of-flight")',
)

# Cytometry file extensions to detect likely raw/analysis data
CYTOMETRY_EXTS = {
    "fcs",   # flow cytometry standard
    "wsp",   # FlowJo workspace
    "lmd",   # cytometry-related (sometimes)
    "csv",
    "tsv",
    "txt",
    "zip",
    "tar",
    "gz",
}

CYTOMETRY_TERMS = [
    "flow cytometry",
    "cytometry",
    "cytof",
    "mass cytometry",
    "facs",
    "fluorescence-activated cell sorting",
    "time-of-flight",
    "fcs",
]

NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
PMID_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{6,9})", re.IGNORECASE)
PMID_TAG_RE = re.compile(r"\bPMID\s*[:#]?\s*(\d{6,9})\b", re.IGNORECASE)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "external-cytometry-discovery/1.0"})


def backoff_sleep(attempt: int) -> None:
    time.sleep(min(10.0, 0.8 * (attempt + 1)))


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


def cytometry_signal(title: str, desc: str, file_names: List[str]) -> Tuple[float, str]:
    """
    Heuristic confidence score [0..1] + reason string.
    """
    t = (title or "").lower()
    d = (desc or "").lower()
    fn = " ".join(file_names).lower()

    reasons = []
    score = 0.0

    # Strong: has .fcs file
    has_fcs = any(file_ext(x) == "fcs" for x in file_names)
    if has_fcs:
        score += 0.6
        reasons.append("has_fcs")

    # Terms in title/desc
    hits_td = [term for term in CYTOMETRY_TERMS if term in t or term in d]
    if hits_td:
        score += 0.3
        reasons.append("terms_in_title_desc:" + ",".join(sorted(set(hits_td))[:5]))

    # Terms in filenames
    hits_fn = [term for term in CYTOMETRY_TERMS if term in fn]
    if hits_fn:
        score += 0.2
        reasons.append("terms_in_filenames")

    # Cap
    score = min(1.0, score)
    return score, " | ".join(reasons)


# -------------------------
# Zenodo
# -------------------------
def zenodo_page_size() -> int:
    """
    Zenodo enforces stricter page sizes for anonymous requests (use 25).
    If token is present, allow up to 100.
    """
    if ZENODO_TOKEN:
        return min(100, max(1, ZENODO_PAGE_SIZE_DEFAULT))
    return min(25, max(1, ZENODO_PAGE_SIZE_DEFAULT))


def zenodo_search_records(query: str, max_pages: int) -> List[Dict[str, Any]]:
    base = "https://zenodo.org/api/records"
    size = zenodo_page_size()
    out: List[Dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "page": page,
            "size": size,
            "sort": "mostrecent",
        }
        # Zenodo tokens are commonly passed as access_token (works for API usage).
        if ZENODO_TOKEN:
            params["access_token"] = ZENODO_TOKEN

        data = http_get_json(base, params=params)
        hits = ((data or {}).get("hits", {}) or {}).get("hits", []) or []
        if not hits:
            break

        out.extend(hits)

        # stop early if fewer than page size returned
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
    # Zenodo also often has "prereserve_doi" or related identifiers; keep simple.
    landing = normalize_text(links.get("html", ""))

    files = hit.get("files", []) or []
    file_names = [normalize_text(f.get("key", "")) for f in files if f.get("key")]
    file_exts = sorted(set([file_ext(n) for n in file_names if file_ext(n)]))
    file_exts_str = ";".join(file_exts)
    file_names_str = ";".join(file_names[:50])

    download_urls = []
    for f in files:
        l = (f.get("links", {}) or {}).get("self") or (f.get("links", {}) or {}).get("download")
        if l:
            download_urls.append(str(l))
    download_urls_str = " | ".join(download_urls[:20])

    text_blob = " ".join([title, desc, keywords_str])
    ncts = extract_ncts(text_blob)
    pmids = extract_pmids(text_blob)

    conf, reason = cytometry_signal(title, desc, file_names)
    tech_guess = "unknown"
    if "cytof" in text_blob.lower() or "mass cytometry" in text_blob.lower():
        tech_guess = "CyTOF/Mass Cytometry"
    elif "flow cytometry" in text_blob.lower() or "facs" in text_blob.lower() or any(file_ext(n) == "fcs" for n in file_names):
        tech_guess = "Flow Cytometry"

    return {
        "source": "zenodo",
        "record_id": str(rec_id),
        "doi": doi,
        "title": title,
        "published_date": pub_date,
        "landing_page_url": landing,
        "description_snippet": desc[:500],
        "keywords": keywords_str,
        "file_names": file_names_str,
        "file_extensions": file_exts_str,
        "download_urls": download_urls_str,
        "nct_ids_extracted": ";".join(ncts),
        "pmid_extracted": ";".join(pmids),
        "technology_guess": tech_guess,
        "cytometry_confidence": conf,
        "cytometry_reason": reason,
    }


# -------------------------
# Figshare
# -------------------------
def figshare_headers() -> Dict[str, str]:
    h = {}
    if FIGSHARE_TOKEN:
        # Figshare supports Authorization: token ACCESS_TOKEN
        h["Authorization"] = f"token {FIGSHARE_TOKEN}"
    return h


def figshare_search_articles(query: str, max_pages: int, page_size: int) -> List[Dict[str, Any]]:
    """
    Uses POST /v2/articles/search.
    We additionally filter to datasets using item_type=3 (dataset).
    """
    base = "https://api.figshare.com/v2/articles/search"
    out: List[Dict[str, Any]] = []
    headers = figshare_headers()

    for page in range(1, max_pages + 1):
        payload = {
            "search_for": query,
            "item_type": 3,               # 3 = Dataset (per Figshare docs)
            "page": page,
            "page_size": page_size,
            "order": "published_date",
            "order_direction": "desc",
        }
        try:
            data = http_post_json(base, payload=payload, headers=headers)
        except Exception:
            # If search narrows too little, Figshare may return max page reached errors (400).
            break

        if not isinstance(data, list) or not data:
            break

        out.extend(data)

        if len(data) < page_size:
            break

    return out


def figshare_get_article(article_id: int) -> Dict[str, Any]:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    return http_get_json(url, headers=figshare_headers())


def parse_figshare_article(article_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    article_summary is typically an ArticlePresenter from search (has id/title/doi/published_date).
    We'll fetch full details to get description/files.
    """
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
    file_exts = sorted(set([file_ext(n) for n in file_names if file_ext(n)]))
    file_exts_str = ";".join(file_exts)
    file_names_str = ";".join(file_names[:50])

    download_urls = []
    for f in files:
        u = f.get("download_url")
        if u:
            download_urls.append(str(u))
    download_urls_str = " | ".join(download_urls[:20])

    text_blob = " ".join([title, desc, tags_str])
    ncts = extract_ncts(text_blob)
    pmids = extract_pmids(text_blob)

    conf, reason = cytometry_signal(title, desc, file_names)

    tech_guess = "unknown"
    if "cytof" in text_blob.lower() or "mass cytometry" in text_blob.lower():
        tech_guess = "CyTOF/Mass Cytometry"
    elif "flow cytometry" in text_blob.lower() or "facs" in text_blob.lower() or any(file_ext(n) == "fcs" for n in file_names):
        tech_guess = "Flow Cytometry"

    return {
        "source": "figshare",
        "record_id": str(aid),
        "doi": doi,
        "title": title,
        "published_date": pub_date,
        "landing_page_url": landing,
        "description_snippet": desc[:500],
        "keywords": tags_str,
        "file_names": file_names_str,
        "file_extensions": file_exts_str,
        "download_urls": download_urls_str,
        "nct_ids_extracted": ";".join(ncts),
        "pmid_extracted": ";".join(pmids),
        "technology_guess": tech_guess,
        "cytometry_confidence": conf,
        "cytometry_reason": reason,
    }


def main() -> None:
    rows: List[Dict[str, Any]] = []

    # --- Zenodo ---
    print(f"[Zenodo] Searching query: {ZENODO_QUERY}")
    zen_hits = zenodo_search_records(ZENODO_QUERY, max_pages=ZENODO_MAX_PAGES)
    print(f"[Zenodo] Hits retrieved: {len(zen_hits)}")
    for h in zen_hits:
        row = parse_zenodo_record(h)
        if row:
            # Filter: keep only if some signal
            if row["cytometry_confidence"] >= 0.3:
                rows.append(row)

    # --- Figshare ---
    fig_page_size = max(1, min(100, FIGSHARE_PAGE_SIZE_DEFAULT))
    print(f"[Figshare] Searching query: {FIGSHARE_QUERY}")
    fig_hits = figshare_search_articles(FIGSHARE_QUERY, max_pages=FIGSHARE_MAX_PAGES, page_size=fig_page_size)
    print(f"[Figshare] Search results retrieved: {len(fig_hits)}")

    # Fetch full details for top N hits to keep runtime predictable
    max_detail = int(os.environ.get("FIGSHARE_MAX_DETAILS", "50"))
    for art in fig_hits[:max_detail]:
        try:
            row = parse_figshare_article(art)
            if row and row.get("cytometry_confidence", 0) >= 0.3:
                rows.append(row)
        except Exception:
            continue

    if not rows:
        print("No external candidates found (or all below confidence threshold).")
        out_path = os.path.join(OUTPUT_DIR, "external_cytometry_candidates.csv")
        pd.DataFrame([]).to_csv(out_path, index=False)
        print(f"Wrote empty {out_path}")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["source", "record_id"]).reset_index(drop=True)

    # Normalize + sort by published_date (newest first where possible)
    df["published_date"] = df["published_date"].fillna("").astype(str)
    df = df.sort_values(by=["published_date", "source"], ascending=[False, True]).reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, "external_cytometry_candidates.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")

    print("\nCounts by source:")
    print(df["source"].value_counts(dropna=False))
    print("\nCounts by technology_guess:")
    print(df["technology_guess"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
