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
    '( "flow cytometry" OR C
