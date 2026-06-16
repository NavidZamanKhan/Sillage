"""
Prepare and clean fra_perfumes.csv for a Flask ML project.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pandas as pd


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

RAW_NAME = "fra_perfumes.csv"
OUT_NAME = "cleaned_perfumes.csv"


def _norm_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower())


def resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Map logical names to actual column names in the dataframe."""
    by_norm = {_norm_col(c): c for c in df.columns}

    def first_match(*needles: str) -> str | None:
        for n in needles:
            n = n.lower()
            for norm, orig in by_norm.items():
                if n in norm or norm == n:
                    return orig
        return None

    return {
        "name": first_match("perfume name", "name", "title", "perfume"),
        "brand": first_match("brand", "house", "designer", "label", "company"),
        "gender": first_match("gender"),
        "top": first_match("top note"),
        "middle": first_match("middle note", "heart note"),
        "base": first_match("base note"),
        "main_accords": first_match("main accord", "accords"),
        "description": first_match("description", "desc"),
        "season": first_match("season", "seasons"),
        "time_of_day": first_match("time of day", "day night", "occasion", "day/night"),
    }


def _parse_notes_from_description(text: str) -> tuple[str, str, str]:
    """Extract Top / Middle / Base from Fragrantica-style descriptions."""
    if not isinstance(text, str) or not text.strip():
        return "", "", ""

    top = middle = base = ""

    m_top = re.search(
        r"(?i)Top notes?\s+(?:are|is)\s+(.+?)(?=\s*(?:[;.]\s*)?(?:Middle|middle)\s+notes?\s+(?:are|is)|$)",
        text,
        re.DOTALL,
    )
    if m_top:
        top = re.sub(r"\s+", " ", m_top.group(1).strip()).strip(" ;.")

    m_mid = re.search(
        r"(?i)Middle notes?\s+(?:are|is)\s+(.+?)(?=\s*(?:[;.]\s*)?(?:Base|base)\s+notes?\s+(?:are|is)|$)",
        text,
        re.DOTALL,
    )
    if m_mid:
        middle = re.sub(r"\s+", " ", m_mid.group(1).strip()).strip(" ;.")

    m_base = re.search(r"(?i)Base notes?\s+(?:are|is)\s+(.+?)(?:\.(?:\s|$)|$)", text, re.DOTALL)
    if not m_base:
        m_base = re.search(r"(?i)Base notes?\s+(?:are|is)\s+(.+)$", text, re.DOTALL)
    if m_base:
        base = re.sub(r"\s+", " ", m_base.group(1).strip()).strip(" ;.")

    return top, middle, base


def _parse_brand_from_description(text: str) -> str:
    """Typical pattern: 'AventusbyCreedis a' -> Creed."""
    if not isinstance(text, str):
        return ""
    m = re.search(r"by([A-Za-z][A-Za-z0-9\s\-'.&]+?)is a\b", text)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())
    return ""


def _accords_to_list(val) -> list[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return [str(x).lower() for x in val]
    s = str(val).strip()
    if not s or s in ("[]", "nan"):
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).lower() for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [s.lower()]


def _combined_notes_text(row: pd.Series, accords_col: str | None) -> str:
    parts = [
        str(row.get("Top Notes", "") or ""),
        str(row.get("Middle Notes", "") or ""),
        str(row.get("Base Notes", "") or ""),
    ]
    if accords_col:
        parts.append(" ".join(_accords_to_list(row.get(accords_col))))
    return " ".join(parts).lower()


# Keyword buckets for inferred season (heuristic)
_SEASON_KEYWORDS = {
    "Summer": (
        "citrus",
        "aquatic",
        "marine",
        "sea",
        "salt",
        "ozonic",
        "tropical",
        "coconut",
        "water",
        "fresh",
        "melon",
        "pineapple",
    ),
    "Winter": (
        "vanilla",
        "amber",
        "cinnamon",
        "spice",
        "oud",
        "incense",
        "balsamic",
        "warm",
        "gourmand",
        "caramel",
        "praline",
    ),
    "Spring": (
        "floral",
        "rose",
        "green",
        "lily",
        "peony",
        "iris",
        "violet",
        "white floral",
        "fresh spicy",
    ),
    "Fall": (
        "woody",
        "patchouli",
        "smoky",
        "earthy",
        "leather",
        "moss",
        "vetiver",
    ),
}

_TIME_KEYWORDS = {
    "Day": ("citrus", "aquatic", "fresh", "green", "marine", "light", "tea"),
    "Night": ("amber", "oud", "vanilla", "musk", "leather", "incense", "sweet"),
}


def infer_season(text: str) -> str:
    if not text.strip():
        return "All Season"
    scores: dict[str, int] = {k: 0 for k in _SEASON_KEYWORDS}
    for season, keys in _SEASON_KEYWORDS.items():
        for k in keys:
            if k in text:
                scores[season] += 1
    best = max(scores.values())
    if best == 0:
        return "All Season"
    winners = [s for s, v in scores.items() if v == best]
    return winners[0] if len(winners) == 1 else "All Season"


def infer_time_of_day(text: str) -> str:
    if not text.strip():
        return "Any"
    day = sum(1 for k in _TIME_KEYWORDS["Day"] if k in text)
    night = sum(1 for k in _TIME_KEYWORDS["Night"] if k in text)
    if day > night:
        return "Day"
    if night > day:
        return "Night"
    return "Any"


def main() -> None:
    _configure_stdout()
    base = Path(__file__).resolve().parent
    raw_path = base / RAW_NAME

    df = pd.read_csv(raw_path)
    raw_count = len(df)

    cols = resolve_columns(df)
    name_col = cols["name"]
    if not name_col:
        raise ValueError("Could not find a perfume name column (e.g. 'Name').")

    # --- Column selection + derived fields ---
    out = pd.DataFrame()
    out["Perfume Name"] = df[name_col]

    if cols["brand"]:
        out["Brand"] = df[cols["brand"]]
    else:
        desc_series = df[cols["description"]] if cols["description"] else pd.Series([""] * len(df))
        out["Brand"] = desc_series.map(_parse_brand_from_description)

    if cols["gender"]:
        out["Gender"] = df[cols["gender"]]
    else:
        out["Gender"] = ""

    if cols["top"] and cols["middle"] and cols["base"]:
        out["Top Notes"] = df[cols["top"]].fillna("").astype(str)
        out["Middle Notes"] = df[cols["middle"]].fillna("").astype(str)
        out["Base Notes"] = df[cols["base"]].fillna("").astype(str)
    else:
        desc_series = df[cols["description"]] if cols["description"] else pd.Series([""] * len(df))
        parsed = desc_series.map(_parse_notes_from_description)
        out["Top Notes"] = parsed.map(lambda t: t[0])
        out["Middle Notes"] = parsed.map(lambda t: t[1])
        out["Base Notes"] = parsed.map(lambda t: t[2])

    acc_src = cols["main_accords"]
    if acc_src:
        out["Main Accords"] = df[acc_src]
    else:
        out["Main Accords"] = ""

    has_season_col = bool(cols["season"])
    has_tod_col = bool(cols["time_of_day"])
    if has_season_col:
        out["Season"] = df[cols["season"]]
    if has_tod_col:
        out["Time of Day"] = df[cols["time_of_day"]]

    # --- Missing favorites: show rows mentioning Aventus / Mademoiselle ---
    for needle in ("Aventus", "Mademoiselle"):
        mask = df[name_col].fillna("").astype(str).str.contains(re.escape(needle), case=False, regex=True)
        hits = df.loc[mask]
        print(f"\n=== Rows whose name contains '{needle}': {len(hits)} match(es) ===")
        if len(hits) == 0:
            print("(none)")
        else:
            with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 80):
                print(hits.to_string())

    # --- Cleanup: drop empty name or completely empty notes pyramid ---
    name_ok = out["Perfume Name"].fillna("").astype(str).str.strip() != ""
    notes_combined = (
        out["Top Notes"].fillna("").astype(str).str.strip()
        + out["Middle Notes"].fillna("").astype(str).str.strip()
        + out["Base Notes"].fillna("").astype(str).str.strip()
    )
    notes_ok = notes_combined != ""
    cleaned = out.loc[name_ok & notes_ok].copy()

    # --- Vibe tagging: Season / Time of Day when missing ---
    vibe_text = cleaned.apply(
        lambda r: _combined_notes_text(r, "Main Accords"),
        axis=1,
    )
    if not has_season_col:
        cleaned["Season"] = vibe_text.map(infer_season)
    if not has_tod_col:
        cleaned["Time of Day"] = vibe_text.map(infer_time_of_day)

    out_path = base / OUT_NAME
    cleaned.to_csv(out_path, index=False)

    print("\n--- Status ---")
    print(f"Raw rows in '{RAW_NAME}':     {raw_count}")
    print(f"Cleaned rows in '{OUT_NAME}': {len(cleaned)}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
