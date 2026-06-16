"""
Service layer: the only module that talks to the ML engine.

Loads artifacts once (lazy, thread-safe) and exposes search_perfumes().
All helper functions ported from the original Flask app.py.
"""

from __future__ import annotations

import math
import re
import sys
import threading
from pathlib import Path
from urllib.parse import urlencode

# ---------------------------------------------------------------------------
# ML directory lives at  backend/ml/
# ---------------------------------------------------------------------------
ML_DIR = Path(__file__).resolve().parent.parent / "ml"

# Make ml/ importable so we can reach train_model
_ml_str = str(ML_DIR)
if _ml_str not in sys.path:
    sys.path.insert(0, _ml_str)

from train_model import get_recommendations, load_artifacts  # noqa: E402

# ---------------------------------------------------------------------------
# One-time lazy model loading
# ---------------------------------------------------------------------------
_MODELS_LOADED = False
_load_lock = threading.Lock()


def _ensure_models_loaded() -> None:
    global _MODELS_LOADED
    if _MODELS_LOADED:
        return
    with _load_lock:
        if _MODELS_LOADED:
            return
        load_artifacts(ML_DIR)
        _MODELS_LOADED = True


# ---------------------------------------------------------------------------
# Helper functions (ported verbatim from Flask app.py)
# ---------------------------------------------------------------------------

def json_safe_value(v):
    """Convert a value to something JSON-serializable."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    return v


def clean_brand(brand) -> str:
    """Normalise brand strings, converting NaN-like values to empty string."""
    if brand is None:
        return ""
    s = str(brand).strip()
    if s.lower() in ("nan", "none", "<na>"):
        return ""
    return s


def image_search_url(brand, perfume_name: str) -> str:
    """Generate a Google image search URL for the perfume bottle."""
    b = clean_brand(brand)
    n = str(perfume_name).strip()
    q = f"{b} {n} perfume bottle".strip()
    return "https://www.google.com/search?" + urlencode({"q": q})


def prepare_results(raw: list[dict]) -> list[dict]:
    """Make every value JSON-safe and attach image_search_url."""
    out = []
    for r in raw:
        row = {}
        for k, v in r.items():
            row[k] = json_safe_value(v)
        row["image_search_url"] = image_search_url(
            r.get("brand"),
            r.get("perfume_name", ""),
        )
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def search_perfumes(
    query: str,
    gender: str | None = None,
    limit: int = 5,
) -> dict:
    """
    Run a perfume search and return a response dict identical to the
    old Flask /search endpoint.
    """
    _ensure_models_loaded()

    query = (query or "").strip()
    if not query:
        return {
            "query": query,
            "gender": gender,
            "limit": limit,
            "results": [],
        }

    raw = get_recommendations(query, gender_filter=gender, k=limit)

    # Fix spacing in perfume names (ported from Flask app.py line 91-96)
    for r in raw:
        pn = r.get("perfume_name")
        if pn is not None:
            r["perfume_name"] = re.sub(
                r"([a-z])(for\s)", r"\1 \2", str(pn), flags=re.IGNORECASE
            )

    return {
        "query": query,
        "gender": gender,
        "limit": limit,
        "results": prepare_results(raw),
    }
