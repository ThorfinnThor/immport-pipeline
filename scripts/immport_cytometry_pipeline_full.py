from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# =========================
# CONFIG (via env in GitHub Actions)
# =========================
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
MAX_RECORDS = int(os.getenv("MAX_RECORDS", "100"))  # hard cap final rows (your request: <= 100)
CLINICAL_TRIAL_ONLY = os.getenv("CLINICAL_TRIAL_ONLY", "1") in ("1", "true", "True", "YES", "yes")

# Gemini fallback
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
GEMINI_MAX_CALLS = int(os.getenv("GEMINI_MAX_CALLS", "50"))
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "60"))

# =========================
# ENDPOINTS
# =========================
IMMPORT_STUDY_SEARCH = "https://www.immport.org/data/query/api/search/study"
IMMPORT_UI_BASE = "https://www.immport.org/data/query/ui"
IMMPORT_LOOKUP_BASE = "https://www.immport.org/data/query/api/lookup"
CTGOV_BASE = "https://clinicaltrials.gov/api/v2"

# PubMed E-utilities
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "immport-cytometry-pipeline/2.0"})

NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)

# =========================
# HTTP HELPERS
# =========================
def _sleep_backoff(i: int) -> None:
    time.sleep(0.8 * (i + 1))


def get_json(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Any:
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                _sleep_backoff(i)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            _sleep_backoff(i)
    raise RuntimeError(f"GET failed: {url} params={params} err={last_err}")


def get_text(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> str:
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                _sleep_backoff(i)
                continue
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            _sleep_backoff(i)
    raise RuntimeError(f"GET TEXT failed: {url} params={params} err={last_err}")


# =========================
# IMMPort: LOOKUP + SEARCH
# =========================
def lookup_exp_measurement_tech() -> List[str]:
    """
    Returns list of assay method terms from the controlled vocabulary table.
    Per ImmPort docs, assayMethod facet values come from lkExpMeasurementTech.
    """
    url = f"{IMMPORT_LOOKUP_BASE}/lkExpMeasurementTech"
    try:
        data = get_json(url, params={"format": "json"})
    except Exception:
        return []

    # The lookup endpoints can return different shapes; try to normalize to a list of strings.
    terms: List[str] = []
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                # best-effort: take any string-ish field that looks like a term
                for k in ("exp_measurement_tech", "measurement_technique", "name", "value", "term"):
                    v = row.get(k)
                    if isinstance(v, str) and v.strip():
                        terms.append(v.strip())
                        break
            elif isinstance(row, str) and row.strip():
                terms.append(row.strip())
    elif isinstance(data, dict):
        # sometimes returned under a key
        for key in ("data", "results", "rows", "items"):
            if key in data and isinstance(data[key], list):
                for row in data[key]:
                    if isinstance(row, dict):
                        for k in ("exp_measurement_tech", "measurement_technique", "name", "value", "term"):
                            v = row.get(k)
                            if isinstance(v, str) and v.strip():
                                terms.append(v.strip())
                                break
                    elif isinstance(row, str) and row.strip():
                        terms.append(row.strip())

    # de-dup preserve order
    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def is_flow_term(t: str) -> bool:
    tl = t.lower()
    return any(x in tl for x in [
        "flow cytometry",
        "facs",
        "fluorescence-activated",
        "cell sorting",  # many flow datasets appear as sorting
        "fluorescence activated cell sorting",
    ])


def is_cytof_term(t: str) -> bool:
    tl = t.lower()
    return any(x in tl for x in [
        "cytof",
        "mass cytometry",
        "time-of-flight cytometry",
        "cytometry by time-of-flight",
        "tof cytometry",
    ])


def search_immport_studies(assay_method: str, page_size: int = 200) -> List[Dict[str, Any]]:
    """
    Paginates ImmPort /search/study results using fromRecord (1-indexed).
    """
    from_record = 1
    out: List[Dict[str, Any]] = []

    while True:
        params = {
            "pageSize": page_size,
            "fromRecord": from_record,
            "assayMethod": assay_method,
            # IMPORTANT: keep your original "clinicalTrial=Y" intent
            **({"clinicalTrial": "Y"} if CLINICAL_TRIAL_ONLY else {}),
            # add release dates so year extraction is more reliable than regex
            "sourceFields": ",".join([
                "study_accession",
                "brief_title",
                "pubmed_id",
                "condition_or_disease",
                "assay_method_count",
                "initial_data_release_date",
                "latest_data_release_date",
            ]),
        }
        data = get_json(IMMPORT_STUDY_SEARCH, params=params)

        hits_block = data.get("hits", {}) or {}
        total_val = (hits_block.get("total", {}) or {}).get("value", None)
        hits = hits_block.get("hits", []) or []

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
    """
    UI metadata endpoints (open) used to pull NCT IDs and file metadata.
    """
    return {
        "summary": get_json(f"{IMMPORT_UI_BASE}/study/summary/{sdy}"),
        "design": get_json(f"{IMMPORT_UI_BASE}/study/design/{sdy}"),
        "studyfile": get_json(f"{IMMPORT_UI_BASE}/study/studyfile/{sdy}"),
    }


def extract_nct_ids(obj: Any) -> List[str]:
    txt = json.dumps(obj, ensure_ascii=False)
    return sorted(set(re.findall(r"\bNCT\d{8}\b", txt)))


def parse_year_from_dates(initial_date: Any, latest_date: Any) -> Optional[int]:
    """
    Prefer ImmPort release dates if present (YYYY-MM-DD).
    """
    for v in (latest_date, initial_date):
        if not v:
            continue
        s = str(v)
        m = re.search(r"\b(19|20)\d{2}\b", s)
        if m:
            return int(m.group(0))
    return None


# =========================
# CT.gov
# =========================
def ctgov_get_study(nct: str) -> Dict[str, Any]:
    return get_json(f"{CTGOV_BASE}/studies/{nct}")


def ctgov_status_whystopped(study_json: Dict[str, Any]) -> Tuple[str, str]:
    proto = (study_json or {}).get("protocolSection", {}) or {}
    status_mod = proto.get("statusModule", {}) or {}
    return str(status_mod.get("overallStatus") or ""), str(status_mod.get("whyStopped") or "")


def ctgov_failure_signal(status: str, why: str) -> Tuple[int, str]:
    status_l = (status or "").upper()
    why_l = (why or "").lower()

    score = 0
    reasons = []

    if status_l in ("TERMINATED", "SUSPENDED", "WITHDRAWN"):
        score += 2
        reasons.append(f"status={status_l}")

    if any(k in why_l for k in ["futility", "lack of efficacy", "inefficacy", "ineffective"]):
        score += 4
        reasons.append(f"whyStopped={why[:120]}")

    return score, " | ".join(reasons)


# =========================
# PubMed (title+abstract)
# =========================
def efetch_pubmed_title_abstract(pubmed_ids: List[str], batch_size: int = 200, sleep: float = 0.34) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i in range(0, len(pubmed_ids), batch_size):
        batch = pubmed_ids[i:i + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        xml_text = get_text(EFETCH, params=params)

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


# =========================
# Gemini fallback (REST)
# =========================
@dataclass
class GeminiResult:
    label: str           # met / not_met / unknown
    confidence: float    # 0..1
    reason: str
    evidence: str
    http_status: int = 0
    http_error: str = ""


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """
    Try json.loads; if fails, attempt to extract first {...} block.
    """
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def gemini_label_from_text(study_text: str) -> GeminiResult:
    """
    Calls Gemini REST API (Developer API) using correct generationConfig fields.
    Uses responseMimeType=application/json to reduce parse failures. :contentReference[oaicite:5]{index=5}
    """
    if not GEMINI_API_KEY:
        return GeminiResult("unknown", 0.0, "no_gemini_key", "", 0, "")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    # Keep prompt compact to reduce cost and avoid truncation.
    prompt = (
        "You are classifying a clinical trial outcome based only on the text below.\n"
        "Task: Decide whether the primary aims/endpoints were met.\n"
        "Return ONLY a JSON object with keys: label, confidence, reason, evidence.\n"
        "label must be one of: met, not_met, unknown.\n"
        "confidence is 0.0 to 1.0.\n"
        "reason: 1-2 sentences.\n"
        "evidence: copy a short phrase from the text that supports your label (or empty if unknown).\n\n"
        "TEXT:\n"
        f"{study_text[:6000]}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
            # Correct field name for REST:
            "responseMimeType": "application/json",
        },
    }

    try:
        r = SESSION.post(url, params={"key": GEMINI_API_KEY}, json=payload, timeout=GEMINI_TIMEOUT)
        status = r.status_code
        if status != 200:
            # capture body to debug (truncate)
            return GeminiResult(
                "unknown",
                0.0,
                "gemini_http_error",
                "",
                http_status=status,
                http_error=r.text[:500],
            )

        data = r.json()
        # extract candidate text
        text = ""
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            # sometimes structured JSON may be inside different shape; dump minimal
            text = json.dumps(data)[:1000]

        parsed = _safe_json_extract(text)
        if not parsed:
            return GeminiResult("unknown", 0.0, "gemini_parse_failed", text[:500], http_status=200, http_error="")

        label = str(parsed.get("label", "unknown")).strip().lower()
        if label not in ("met", "not_met", "unknown"):
            label = "unknown"

        conf = parsed.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        conf_f = max(0.0, min(1.0, conf_f))

        reason = str(parsed.get("reason", "")).strip()[:500]
        evidence = str(parsed.get("evidence", "")).strip()[:500]

        return GeminiResult(label, conf_f, reason or "ok", evidence, http_status=200, http_error="")

    except Exception as e:
        return GeminiResult("unknown", 0.0, "gemini_exception", "", http_status=0, http_error=str(e)[:500])


# =========================
# Fusion + Sorting
# =========================
def fuse_labels(pub_label: str, pub_score: int, ct_score: int, gem: GeminiResult) -> Tuple[str, str]:
    """
    Conservative fusion:
    1) Strong PubMed signal wins.
    2) Gemini wins only if confidence >= 0.65 and label is met/not_met.
    3) Strong CT.gov futility/inefficacy => not_met.
    Else unknown.
    """
    if pub_label == "not_met" and pub_score >= 4:
        return "not_met", "high: PubMed text"
    if pub_label == "met" and pub_score <= -4:
        return "met", "high: PubMed text"

    if gem.label in ("met", "not_met") and gem.confidence >= 0.65:
        return gem.label, f"med: Gemini ({gem.confidence:.2f})"

    if ct_score >= 6:
        return "not_met", "med-high: CT.gov futility/inefficacy"
    if ct_score >= 2:
        return "not_met_candidate", "medium: CT.gov terminated/suspended (reason unclear)"

    return "unknown", "low: no deterministic signal"


def tech_group(method: str) -> str:
    ml = (method or "").lower()
    if "flow" in ml or "facs" in ml or "sorting" in ml:
        return "Flow Cytometry"
    if "cytof" in ml or "mass" in ml or "time-of-flight" in ml or "tof" in ml:
        return "CyTOF / Mass Cytometry"
    return "Other"


def tech_sort_key(group: str) -> int:
    if group == "Flow Cytometry":
        return 0
    if group == "CyTOF / Mass Cytometry":
        return 1
    return 9


# =========================
# MAIN
# =========================
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Build assayMethod list from vocabulary + synonyms (more complete)
    vocab = lookup_exp_measurement_tech()
    flow_terms = [t for t in vocab if is_flow_term(t)]
    cytof_terms = [t for t in vocab if is_cytof_term(t)]

    # Hard fallback (in case lookup endpoint fails or is incomplete)
    flow_terms += [
        "Flow Cytometry",
        "FACS",
        "Fluorescence-activated cell sorting",
        "Cell sorting",
    ]
    cytof_terms += [
        "CyTOF",
        "Mass Cytometry",
        "Mass cytometry (CyTOF)",
        "Cytometry by time-of-flight",
        "Cytometry by Time-of-Flight",
        "Time-of-flight cytometry",
    ]

    # de-dup preserve order
    def dedup(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            x = x.strip()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    flow_terms = dedup(flow_terms)
    cytof_terms = dedup(cytof_terms)

    print(f"Using Flow assay methods ({len(flow_terms)}): {flow_terms[:12]}{'...' if len(flow_terms) > 12 else ''}")
    print(f"Using CyTOF assay methods ({len(cytof_terms)}): {cytof_terms[:12]}{'...' if len(cytof_terms) > 12 else ''}")

    # 2) Query studies for each assay method value
    rows: List[Dict[str, Any]] = []

    def add_studies_for_terms(terms: List[str], label: str) -> None:
        for term in terms:
            try:
                studies = search_immport_studies(term, page_size=200)
            except Exception as e:
                print(f"[WARN] search failed for assayMethod={term}: {e}")
                continue

            print(f"{label} via assayMethod='{term}': found {len(studies)} studies")

            for s in studies:
                sdy = s.get("study_accession")
                if not sdy:
                    continue

                # Enrich with UI bundle for NCT IDs
                try:
                    ui = immport_ui_bundle(sdy)
                    ncts = extract_nct_ids(ui)
                    ui_error = ""
                    has_studyfile = True
                except Exception as e:
                    ui = {"error": str(e)}
                    ncts = []
                    ui_error = str(e)
                    has_studyfile = False

                year = parse_year_from_dates(s.get("initial_data_release_date"), s.get("latest_data_release_date"))

                rows.append({
                    "study_accession": sdy,
                    "assay_method": term,
                    "assay_family": label,
                    "brief_title": s.get("brief_title", ""),
                    "pubmed_id": ";".join(s.get("pubmed_id", [])) if isinstance(s.get("pubmed_id"), list) else (s.get("pubmed_id") or ""),
                    "condition_or_disease": ";".join(s.get("condition_or_disease", [])) if isinstance(s.get("condition_or_disease"), list) else (s.get("condition_or_disease") or ""),
                    "nct_ids_merged": ";".join(ncts),
                    "study_year": year if year is not None else "",
                    "initial_data_release_date": s.get("initial_data_release_date", ""),
                    "latest_data_release_date": s.get("latest_data_release_date", ""),
                    "ui_bundle_has_studyfile": bool(has_studyfile),
                    "ui_bundle_error": ui_error,
                })

    add_studies_for_terms(flow_terms, "Flow Cytometry")
    add_studies_for_terms(cytof_terms, "CyTOF / Mass Cytometry")

    df = pd.DataFrame(rows).drop_duplicates(subset=["study_accession"])
    # cap to MAX_RECORDS (keep most recent if possible)
    df["study_year_num"] = pd.to_numeric(df["study_year"], errors="coerce")
    df = df.sort_values(["study_year_num"], ascending=False).head(MAX_RECORDS).copy()

    base_csv = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates_base.csv")
    df.to_csv(base_csv, index=False)
    print(f"Wrote {base_csv} ({len(df)} rows)")

    # 3) CT.gov scoring
    ct_statuses, ct_whys, ct_scores, ct_reasons = [], [], [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="CT.gov status/why"):
        ncts = [x.strip() for x in str(r.get("nct_ids_merged", "")).split(";") if x.strip().upper().startswith("NCT")]
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
                cj = ctgov_get_study(nct.upper())
                status, why = ctgov_status_whystopped(cj)
                sc, rsn = ctgov_failure_signal(status, why)
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

    # 4) PubMed scoring
    df["pubmed_id"] = df["pubmed_id"].fillna("").astype(str)
    unique_pmids = sorted(set(
        pmid.strip()
        for cell in df["pubmed_id"].tolist()
        for pmid in cell.split(";")
        if pmid.strip().isdigit()
    ))

    print(f"Unique PubMed IDs: {len(unique_pmids)}")
    pub_texts = efetch_pubmed_title_abstract(unique_pmids, batch_size=200) if unique_pmids else {}

    pub_labels, pub_scores, pub_evidence, pub_used_text = [], [], [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="PubMed scoring"):
        pmids = [p.strip() for p in str(r["pubmed_id"]).split(";") if p.strip().isdigit()]
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
        pub_used_text.append(text)

    df["pubmed_label_scored"] = pub_labels
    df["pubmed_score"] = pub_scores
    df["pubmed_evidence_patterns"] = pub_evidence

    # 5) Gemini fallback only when still unknown and only up to GEMINI_MAX_CALLS
    gem_labels, gem_conf, gem_reason, gem_evidence, gem_http, gem_http_err = [], [], [], [], [], []
    gem_calls = 0

    for i, r in tqdm(df.iterrows(), total=len(df), desc="Gemini fallback"):
        pub_lab = str(r.get("pubmed_label_scored", "unknown"))
        pub_sc = int(r.get("pubmed_score", 0) or 0)
        ct_sc = int(r.get("ctgov_failure_score", 0) or 0)

        # If deterministic already, skip Gemini
        deterministic = (pub_lab in ("met", "not_met") and abs(pub_sc) >= 4) or (ct_sc >= 6)
        if deterministic or not GEMINI_API_KEY or gem_calls >= GEMINI_MAX_CALLS:
            gem_labels.append("unknown")
            gem_conf.append(0.0)
            gem_reason.append("skipped")
            gem_evidence.append("")
            gem_http.append(0)
            gem_http_err.append("")
            continue

        # Only run Gemini if still ambiguous
        # Provide title+abstract (best-effort)
        text = pub_used_text[df.index.get_loc(i)] if df.index.get_loc(i) < len(pub_used_text) else ""
        if not text:
            gem_labels.append("unknown")
            gem_conf.append(0.0)
            gem_reason.append("no_pubmed_text")
            gem_evidence.append("")
            gem_http.append(0)
            gem_http_err.append("")
            continue

        gr = gemini_label_from_text(text)
        gem_calls += 1

        gem_labels.append(gr.label)
        gem_conf.append(gr.confidence)
        gem_reason.append(gr.reason)
        gem_evidence.append(gr.evidence)
        gem_http.append(gr.http_status)
        gem_http_err.append(gr.http_error)

    df["gemini_label"] = gem_labels
    df["gemini_confidence"] = gem_conf
    df["gemini_reason"] = gem_reason
    df["gemini_evidence"] = gem_evidence
    df["gemini_http_status"] = gem_http
    df["gemini_http_error"] = gem_http_err

    # 6) Final fusion
    final_label, final_conf = [], []
    for _, r in df.iterrows():
        gem = GeminiResult(
            label=str(r.get("gemini_label", "unknown")),
            confidence=float(r.get("gemini_confidence", 0.0) or 0.0),
            reason=str(r.get("gemini_reason", "")),
            evidence=str(r.get("gemini_evidence", "")),
            http_status=int(r.get("gemini_http_status", 0) or 0),
            http_error=str(r.get("gemini_http_error", "")),
        )
        lab, conf = fuse_labels(
            pub_label=str(r.get("pubmed_label_scored", "unknown")),
            pub_score=int(r.get("pubmed_score", 0) or 0),
            ct_score=int(r.get("ctgov_failure_score", 0) or 0),
            gem=gem,
        )
        final_label.append(lab)
        final_conf.append(conf)

    df["final_outcome_label"] = final_label
    df["final_confidence"] = final_conf

    # 7) Sorting: year + technology (Flow first, then CyTOF/Mass)
    df["technology_group"] = df["assay_family"].apply(lambda x: x if x else tech_group(""))
    df["tech_sort"] = df["technology_group"].apply(tech_sort_key)
    df["study_year_num"] = pd.to_numeric(df["study_year"], errors="coerce")

    # Sort: tech first, then year desc (most recent first), then title
    df = df.sort_values(
        by=["tech_sort", "study_year_num", "brief_title"],
        ascending=[True, False, True],
    ).drop(columns=["tech_sort"])

    # 8) Write final outputs
    full_csv = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates_full_ranked.csv")
    df.to_csv(full_csv, index=False)
    print(f"Wrote {full_csv} ({len(df)} rows)")

    print("\nFinal label counts:")
    print(df["final_outcome_label"].value_counts(dropna=False))

    print("\nStudies by technology group:")
    print(df["technology_group"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
