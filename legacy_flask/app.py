"""
Flask app: instant search API backed by TF-IDF + KNN recommendations.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, jsonify, render_template, request

from train_model import get_recommendations, load_artifacts

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)


def _json_safe_value(v):
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


def _clean_brand(brand) -> str:
    if brand is None:
        return ""
    s = str(brand).strip()
    if s.lower() in ("nan", "none", "<na>"):
        return ""
    return s


def _image_search_url(brand, perfume_name: str) -> str:
    b = _clean_brand(brand)
    n = str(perfume_name).strip()
    q = f"{b} {n} perfume bottle".strip()
    return "https://www.google.com/search?" + urlencode({"q": q})


def _prepare_results(raw: list[dict]) -> list[dict]:
    out = []
    for r in raw:
        row = {}
        for k, v in r.items():
            row[k] = _json_safe_value(v)
        row["image_search_url"] = _image_search_url(
            r.get("brand"),
            r.get("perfume_name", ""),
        )
        out.append(row)
    return out


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
def search():
    query = (request.args.get("query") or "").strip()
    gender = request.args.get("gender")
    if gender is not None:
        gender = gender.strip() or None

    limit = request.args.get("limit", default=5, type=int)
    if limit is None or limit < 1:
        limit = 5
    limit = min(limit, 100)

    if not query:
        return jsonify(
            {"query": query, "gender": gender, "limit": limit, "results": []}
        )

    try:
        raw = get_recommendations(query, gender_filter=gender, k=limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    for r in raw:
        pn = r.get("perfume_name")
        if pn is not None:
            r["perfume_name"] = re.sub(
                r"([a-z])(for\s)", r"\1 \2", str(pn), flags=re.IGNORECASE
            )

    return jsonify(
        {
            "query": query,
            "gender": gender,
            "limit": limit,
            "results": _prepare_results(raw),
        }
    )


def _load_models() -> None:
    load_artifacts(BASE_DIR)


_load_models()

if __name__ == "__main__":
    app.run(debug=True)
