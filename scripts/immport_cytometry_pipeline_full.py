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

# Gemini REST endpoint (official): https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=...
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "immport-cytometry-pipeline-full/3.0"})

NCT_RE = re.compile(r"\bNCT\d{8}\b")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gemini config via env
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite").strip()
GEMINI_MAX_CALLS = int(os.environ.get("GEMINI_MAX_CALLS", "50"))

# -----------------------------
# HTTP helpers
# -----------------------------
def _sleep_backoff(i: int) -> None:
    time.sleep(0.8 * (i + 1))


def get_json(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Any:
    last_err = None
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
    raise RuntimeError(f"GET JSON failed: {url} params={params} err={last_err}")


def get_text(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> str:
    last_err = None
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


def post_json(url: str, payload: Dict[str, Any], params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Any:
    last_err = None
    for i in range(retries):
        try:
            r = SESSION.post(url, params=params, json=payload, timeout=90)
            if r.status_code in (429, 500, 502, 503, 504):
                _sleep_backoff(i)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            _sleep_backoff(i)
    raise RuntimeError(f"POST JSON failed: {url} params={params} err={last_err}")


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
    counts: Dict[str, int] = {}
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


def search_immport_studies(assay_method: str, page_size: int = 200) -> List[Dict[str, Any]]:
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
def ctgov_get_study(nct: str) -> Dict[str, Any]:
    return get_json(f"{CTGOV_BASE}/studies/{nct}")


def ctgov_status_whystopped(study_json: Dict[str, Any]) -> Tuple[str, str]:
    proto = (study_json or {}).get("protocolSection", {}) or {}
    status_mod = proto.get("statusModule", {}) or {}
    return str(status_mod.get("overallStatus") or ""), str(status_mod.get("whyStopped") or "")


def ctgov_primary_outcomes(study_json: Dict[str, Any], limit: int = 3) -> List[str]:
    proto = (study_json or {}).get("protocolSection", {}) or {}
    out_mod = proto.get("outcomesModule", {}) or {}
    prim = out_mod.get("primaryOutcomes") or []
    out: List[str] = []
    for item in prim[:limit]:
        if not isinstance(item, dict):
            continue
        measure = (item.get("measure") or "").strip()
        timeframe = (item.get("timeFrame") or "").strip()
        desc = (item.get("description") or "").strip()
        line = " | ".join([x for x in [measure, timeframe, desc] if x])
        if line:
            out.append(line[:300])
    return out


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
# PubMed (Option B)
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
# Gemini (Option C)
# -----------------------------
def _extract_first_candidate_text(resp_json: Dict[str, Any]) -> str:
    # Typical response: candidates[0].content.parts[0].text
    cands = resp_json.get("candidates") or []
    if not cands:
        return ""
    content = (cands[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        return ""
    txt = (parts[0] or {}).get("text") or ""
    return str(txt)


def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    # Often it's already pure JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fallback: extract first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def gemini_label_primary_endpoint(
    *,
    title_abstract: str,
    nct: str,
    ct_status: str,
    ct_why: str,
    ct_primary_outcomes: List[str],
) -> Tuple[str, float, str, str]:
    """
    Returns: (label, confidence, reason, evidence_quote)
    label in {met, not_met, unknown}
    """
    if not GEMINI_API_KEY:
        return ("", 0.0, "no_api_key", "")

    # Strict instruction: do NOT guess.
    prompt = f"""
You are a cautious clinical trials analyst.
Task: Decide if the trial MET its PRIMARY ENDPOINT(S), based ONLY on the text provided below.
Rules:
- If the text does not explicitly state primary endpoint met/not met (or equivalent), output "unknown".
- Do NOT infer from secondary endpoints, biomarkers, or general positive/negative language.
- Do NOT guess. If unclear, output "unknown".
Output MUST be valid JSON with keys: label, confidence, reason, evidence.
label must be one of: "met", "not_met", "unknown".
confidence must be a number from 0 to 1.
reason: <= 200 characters.
evidence: quote <= 25 words copied from the provided text.

TRIAL CONTEXT
NCT: {nct or "unknown"}
CT.gov overallStatus: {ct_status or "unknown"}
CT.gov whyStopped: {ct_why or "unknown"}
CT.gov primary outcomes:
{chr(10).join(["- " + x for x in ct_primary_outcomes]) if ct_primary_outcomes else "- unknown"}

PUBMED TITLE+ABSTRACT
{title_abstract[:12000]}
""".strip()

    url = f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateContent"
    params = {"key": GEMINI_API_KEY}

    # Use JSON-mode generationConfig (REST uses snake_case, per docs)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 256,
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "confidence": {"type": "NUMBER"},
                    "reason": {"type": "STRING"},
                    "evidence": {"type": "STRING"},
                },
                "required": ["label", "confidence"],
            },
        },
    }

    resp = post_json(url, payload=payload, params=params, retries=3)
    txt = _extract_first_candidate_text(resp)
    obj = _parse_json_from_text(txt)
    if not obj:
        return ("unknown", 0.0, "gemini_parse_failed", (txt or "")[:200])

    label = str(obj.get("label", "")).strip().lower()
    conf = obj.get("confidence", 0.0)
    reason = str(obj.get("reason", "")).strip()
    evidence = str(obj.get("evidence", "")).strip()

    # sanitize
    if label not in ("met", "not_met", "unknown"):
        label = "unknown"
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return (label, conf, reason[:220], evidence[:220])


# -----------------------------
# Fusion + sorting
# -----------------------------
def fuse_labels(pub_label: str, pub_score: int, ct_score: int) -> Tuple[str, str]:
    if pub_label == "not_met" and pub_score >= 4:
        return "not_met", "high: PubMed text"
    if pub_label == "met" and pub_score <= -4:
        return "met", "high: PubMed text"

    if ct_score >= 6:
        return "not_met", "med-high: CT.gov futility/inefficacy"
    if ct_score >= 2:
        return "not_met_candidate", "medium: CT.gov terminated/suspended (reason unclear)"

    return "unknown", "low: no deterministic signal"


def tech_sort_key(technology: str) -> int:
    s = (technology or "").lower()
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
    # More complete synonym probing (ImmPort often requires exact strings, so we probe)
    FLOW_CANDIDATES = [
        "Flow Cytometry",
        "Flow cytometry",
        "Flow Cytometry Assay",
        "Flow cytometry assay",
        "FACS",
        "Fluorescence-activated cell sorting",
        "Fluorescence Activated Cell Sorting",
        "Fluorescence activated cell sorting (FACS)",
        "Immunophenotyping (flow cytometry)",
        "Immunophenotyping",
    ]

    CYTOF_CANDIDATES = [
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

    print("Probing Flow Cytometry assayMethod vocabulary…")
    flow_counts = probe_assay_methods(FLOW_CANDIDATES)
    for k, v in flow_counts.items():
        print(f"  {k!r}: {v}")
    WORKING_FLOW = [k for k, v in flow_counts.items() if v and v > 0]
    if not WORKING_FLOW:
        WORKING_FLOW = ["Flow Cytometry"]
        print("WARNING: No Flow synonyms returned hits; falling back to 'Flow Cytometry' only.")
    else:
        print(f"Using Flow assay methods: {WORKING_FLOW}")

    print("\nProbing CyTOF/Mass cytometry assayMethod vocabulary…")
    cy_counts = probe_assay_methods(CYTOF_CANDIDATES)
    for k, v in cy_counts.items():
        print(f"  {k!r}: {v}")
    WORKING_CYTOF = [k for k, v in cy_counts.items() if v and v > 0]
    if WORKING_CYTOF:
        print(f"Using CyTOF assay methods: {WORKING_CYTOF}")
    else:
        print("WARNING: No CyTOF methods returned hits. Proceeding without CyTOF.")

    search_plan: List[Tuple[str, str]] = []
    for m in WORKING_FLOW:
        search_plan.append((m, "Flow Cytometry"))
    for m in WORKING_CYTOF:
        search_plan.append((m, "CyTOF/Mass Cytometry"))

    # -----------------------------
    # 1) ImmPort collection
    # -----------------------------
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
                    "technology": technology,
                    "assay_method_raw_query": assay_method_query,
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

    df = pd.DataFrame(all_rows).reset_index(drop=True)

    # -----------------------------
    # 2) PubMed batch fetch + scoring
    # -----------------------------
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

    pub_labels, pub_scores, pub_evidence, used_pmids, pub_text_used = [], [], [], [], []
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
        used_pmids.append(used)
        pub_text_used.append(text)

    df["pubmed_label_scored"] = pub_labels
    df["pubmed_score"] = pub_scores
    df["pubmed_evidence_patterns"] = pub_evidence
    df["pubmed_pmid_used"] = used_pmids

    # -----------------------------
    # 3) CT.gov scoring (best-of up to 5 NCTs)
    # -----------------------------
    ct_best_nct, ct_statuses, ct_whys, ct_scores, ct_reasons, ct_primary = [], [], [], [], [], []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="CT.gov status/why"):
        ncts = [x for x in str(r["nct_ids"]).split(";") if x.startswith("NCT")]
        if not ncts:
            ct_best_nct.append("")
            ct_statuses.append("")
            ct_whys.append("")
            ct_scores.append(0)
            ct_reasons.append("")
            ct_primary.append("")
            continue

        best_score = 0
        best_nct = ""
        best_status = ""
        best_why = ""
        best_reason = ""
        best_primary = []

        for nct in ncts[:5]:
            try:
                cj = ctgov_get_study(nct)
                status, why = ctgov_status_whystopped(cj)
                prim = ctgov_primary_outcomes(cj, limit=3)
                sc, rsn = ctgov_failure_score(status, why)

                if sc > best_score:
                    best_score = sc
                    best_nct = nct
                    best_status = status
                    best_why = why
                    best_reason = rsn
                    best_primary = prim
            except Exception:
                continue

        ct_best_nct.append(best_nct)
        ct_statuses.append(best_status)
        ct_whys.append(best_why)
        ct_scores.append(best_score)
        ct_reasons.append(f"{best_nct} | {best_reason}" if best_nct or best_reason else "")
        ct_primary.append("; ".join(best_primary))

    df["ctgov_best_nct"] = ct_best_nct
    df["ctgov_best_status"] = ct_statuses
    df["ctgov_best_whyStopped"] = ct_whys
    df["ctgov_failure_score"] = ct_scores
    df["ctgov_failure_reason"] = ct_reasons
    df["ctgov_primary_outcomes"] = ct_primary

    # -----------------------------
    # 4) Fuse A+B baseline
    # -----------------------------
    base_label, base_conf = [], []
    for _, r in df.iterrows():
        lab, conf = fuse_labels(
            pub_label=str(r["pubmed_label_scored"]),
            pub_score=int(r["pubmed_score"]),
            ct_score=int(r["ctgov_failure_score"]),
        )
        base_label.append(lab)
        base_conf.append(conf)

    df["baseline_outcome_label"] = base_label
    df["baseline_confidence"] = base_conf

    # -----------------------------
    # 5) Gemini labeling (Option C) ONLY when baseline is weak
    # -----------------------------
    gem_labels, gem_confs, gem_reasons, gem_evidence = [], [], [], []

    gem_calls = 0
    for i, r in tqdm(df.iterrows(), total=len(df), desc="Gemini labeling (selective)"):
        baseline = str(r["baseline_outcome_label"])
        pub_score = int(r["pubmed_score"])

        # Decide whether to call Gemini:
        # - baseline unknown or not_met_candidate OR
        # - pub_score is weak/ambiguous (between -3 and +3)
        need_gemini = baseline in ("unknown", "not_met_candidate") or (-3 <= pub_score <= 3)

        if not GEMINI_API_KEY or not need_gemini or gem_calls >= GEMINI_MAX_CALLS:
            gem_labels.append("")
            gem_confs.append(0.0)
            gem_reasons.append("skipped")
            gem_evidence.append("")
            continue

        title_abs = pub_text_used[i] if i < len(pub_text_used) else ""
        if not title_abs:
            gem_labels.append("")
            gem_confs.append(0.0)
            gem_reasons.append("no_pubmed_text")
            gem_evidence.append("")
            continue

        nct = str(r.get("ctgov_best_nct", "") or "")
        ct_status = str(r.get("ctgov_best_status", "") or "")
        ct_why = str(r.get("ctgov_best_whyStopped", "") or "")
        prim = str(r.get("ctgov_primary_outcomes", "") or "")
        prim_list = [p.strip() for p in prim.split(";") if p.strip()]

        try:
            lab, conf, reason, evidence = gemini_label_primary_endpoint(
                title_abstract=title_abs,
                nct=nct,
                ct_status=ct_status,
                ct_why=ct_why,
                ct_primary_outcomes=prim_list,
            )
            gem_labels.append(lab)
            gem_confs.append(conf)
            gem_reasons.append(reason)
            gem_evidence.append(evidence)
            gem_calls += 1
        except Exception as e:
            gem_labels.append("unknown")
            gem_confs.append(0.0)
            gem_reasons.append(f"gemini_error:{str(e)[:120]}")
            gem_evidence.append("")

    df["gemini_label"] = gem_labels
    df["gemini_confidence"] = gem_confs
    df["gemini_reason"] = gem_reasons
    df["gemini_evidence"] = gem_evidence

    # -----------------------------
    # 6) Final decision: allow Gemini to override ONLY with high confidence
    # -----------------------------
    final_labels, final_conf = [], []
    for _, r in df.iterrows():
        base = str(r["baseline_outcome_label"])
        g_lab = str(r.get("gemini_label", "") or "").strip()
        g_conf = float(r.get("gemini_confidence", 0.0) or 0.0)

        # Only trust Gemini when it is confident and provides met/not_met
        if g_lab in ("met", "not_met") and g_conf >= 0.75:
            final_labels.append(g_lab)
            final_conf.append(f"gemini:{g_conf:.2f}")
        else:
            final_labels.append(base)
            final_conf.append(str(r["baseline_confidence"]))

    df["final_outcome_label"] = final_labels
    df["final_confidence"] = final_conf

    # -----------------------------
    # 7) Sort ranked output: year desc, tech order (Flow first), label rank
    # -----------------------------
    df["year_int"] = df["study_year"].apply(safe_int_year)
    df["tech_order"] = df["technology"].apply(tech_sort_key)

    label_rank = {"not_met": 0, "not_met_candidate": 1, "unknown": 2, "met": 3}
    df["label_rank"] = df["final_outcome_label"].map(label_rank).fillna(99).astype(int)

    df_ranked = df.sort_values(
        by=["year_int", "tech_order", "label_rank", "study_accession"],
        ascending=[False, True, True, True],
    ).drop(columns=["year_int", "tech_order", "label_rank"]).reset_index(drop=True)

    # -----------------------------
    # 8) Outputs (keep the ranked filename for your email attachment)
    # -----------------------------
    out_full = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates_full.csv")
    out_ranked = os.path.join(OUTPUT_DIR, "immport_cytometry_candidates_full_ranked.csv")

    df.to_csv(out_full, index=False)
    df_ranked.to_csv(out_ranked, index=False)

    print(f"\nGemini calls used: {gem_calls}/{GEMINI_MAX_CALLS}")
    print(f"Wrote: {out_full}")
    print(f"Wrote: {out_ranked}")
    print("\nFinal label counts:")
    print(df_ranked["final_outcome_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
