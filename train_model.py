"""
Train TF-IDF + NearestNeighbors (cosine) for perfume similarity and vibe search.
Adds semantic query parsing + intent-aware reranking on top of KNN retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_CSV = "cleaned_perfumes.csv"
ARTIFACT_KNN = "perfumes_knn.joblib"
ARTIFACT_TFIDF = "perfumes_tfidf.joblib"
ARTIFACT_DF = "perfumes_df.joblib"

# Filled when training or after load_artifacts()
PERFUME_DF: pd.DataFrame | None = None
NN_MODEL: NearestNeighbors | None = None
TFIDF_VECTORIZER: TfidfVectorizer | None = None

VIP_BRANDS = {
    "Amouage", "Creed", "Maison Francis Kurkdjian", "MFK",
    "Parfums de Marly", "Initio", "Xerjoff",
    "Louis Vuitton", "LV",
    "Tom Ford", "Yves Saint Laurent", "YSL",
    "Giorgio Armani", "Armani",
    "Dior", "Chanel",
    "Mancera", "Montale",
    "Byredo", "Le Labo",
    "Diptyque", "Maison Margiela",
    "Frederic Malle", "Penhaligon's"
}

NOTE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "mango": ("mango", "tropical", "juicy", "fruity", "sweet"),
    "pineapple": ("pineapple", "tropical", "fruity", "juicy", "bright"),
    "apple": ("apple", "fruity", "crisp", "fresh", "juicy"),
    "pear": ("pear", "fruity", "juicy", "fresh", "soft"),
    "peach": ("peach", "fruity", "sweet", "juicy", "velvety"),
    "plum": ("plum", "dark fruity", "sweet", "rich", "fruity"),
    "cherry": ("cherry", "fruity", "sweet", "dark", "liqueur"),
    "berry": ("berry", "berries", "fruity", "sweet", "juicy"),
    "strawberry": ("strawberry", "berry", "fruity", "sweet", "juicy"),
    "raspberry": ("raspberry", "berry", "fruity", "sweet", "tart"),
    "blackcurrant": ("blackcurrant", "cassis", "fruity", "green", "tart"),
    "lychee": ("lychee", "litchi", "fruity", "juicy", "sweet"),
    "fig": ("fig", "green", "milky", "creamy", "fruity"),
    "coconut": ("coconut", "creamy", "milky", "sweet", "tropical"),
    "melon": ("melon", "watery", "aquatic", "fresh", "fruity"),
    "banana": ("banana", "tropical", "sweet", "creamy", "fruity"),
    "bergamot": ("bergamot", "citrus", "fresh", "bright", "sparkling"),
    "lemon": ("lemon", "citrus", "fresh", "zesty", "bright"),
    "orange": ("orange", "citrus", "fresh", "sweet citrus", "bright"),
    "mandarin": ("mandarin", "citrus", "juicy", "fresh", "bright"),
    "grapefruit": ("grapefruit", "citrus", "fresh", "sharp", "bitter"),
    "lime": ("lime", "citrus", "fresh", "zesty", "green"),
    "neroli": ("neroli", "citrus floral", "clean", "fresh", "white floral"),
    "petitgrain": ("petitgrain", "green citrus", "fresh", "aromatic", "clean"),
    "yuzu": ("yuzu", "citrus", "fresh", "zesty", "bright"),
    "rose": ("rose", "floral", "romantic", "soft", "elegant"),
    "jasmine": ("jasmine", "white floral", "floral", "sweet floral", "radiant"),
    "tuberose": ("tuberose", "white floral", "creamy floral", "loud floral", "sweet"),
    "orange blossom": ("orange blossom", "white floral", "sweet floral", "clean floral", "radiant"),
    "ylang": ("ylang", "ylang ylang", "floral", "creamy floral", "sweet floral"),
    "lavender": ("lavender", "aromatic", "clean", "fresh", "herbal"),
    "iris": ("iris", "powdery", "lipstick", "elegant", "soft"),
    "violet": ("violet", "powdery", "floral", "soft", "vintage"),
    "lily": ("lily", "floral", "fresh floral", "clean floral", "watery floral"),
    "gardenia": ("gardenia", "white floral", "creamy floral", "sweet floral", "rich"),
    "peony": ("peony", "floral", "airy floral", "pink floral", "soft"),
    "freesia": ("freesia", "fresh floral", "clean floral", "airy", "soft"),
    "magnolia": ("magnolia", "floral", "creamy floral", "soft", "airy"),
    "lotus": ("lotus", "watery floral", "aquatic floral", "soft", "clean"),
    "vanilla": ("vanilla", "sweet", "gourmand", "creamy", "warm"),
    "tonka": ("tonka", "tonka bean", "sweet", "warm", "gourmand"),
    "caramel": ("caramel", "sweet", "gourmand", "warm", "sticky sweet"),
    "chocolate": ("chocolate", "cacao", "gourmand", "sweet", "dark sweet"),
    "coffee": ("coffee", "dark", "roasted", "gourmand", "bitter sweet"),
    "honey": ("honey", "sweet", "warm", "ambery", "rich"),
    "almond": ("almond", "nutty", "sweet", "creamy", "powdery"),
    "lactonic": ("lactonic", "milky", "creamy", "smooth", "soft"),
    "gourmand": ("gourmand", "sweet", "dessert", "edible", "creamy"),
    "oud": ("oud", "agarwood", "woody", "dark", "resinous"),
    "wood": ("wood", "woody", "dry wood", "smooth wood", "warm wood"),
    "sandalwood": ("sandalwood", "creamy wood", "woody", "soft wood", "smooth"),
    "cedar": ("cedar", "woody", "dry wood", "clean wood", "aromatic wood"),
    "vetiver": ("vetiver", "earthy", "green", "woody", "dry"),
    "patchouli": ("patchouli", "earthy", "dark", "woody", "rich"),
    "oakmoss": ("oakmoss", "mossy", "green", "earthy", "classic"),
    "leather": ("leather", "dark", "smoky", "animalic", "bold"),
    "tobacco": ("tobacco", "warm", "sweet tobacco", "smoky", "rich"),
    "smoke": ("smoke", "smoky", "dark", "burnt", "incense"),
    "incense": ("incense", "resinous", "smoky", "churchy", "dark"),
    "amber": ("amber", "ambery", "warm", "resinous", "rich"),
    "benzoin": ("benzoin", "resinous", "vanillic", "warm", "balsamic"),
    "labdanum": ("labdanum", "ambery", "resinous", "dark", "balsamic"),
    "myrrh": ("myrrh", "resinous", "dark", "incense", "balsamic"),
    "frankincense": ("frankincense", "incense", "resinous", "churchy", "smoky"),
    "musk": ("musk", "musky", "clean musk", "skin scent", "soft"),
    "soapy": ("soapy", "clean", "fresh laundry", "detergent", "white musk"),
    "clean": ("clean", "soapy", "fresh", "laundry", "musky"),
    "powdery": ("powdery", "soft", "cosmetic", "iris", "violet"),
    "airy": ("airy", "light", "soft", "clean", "fresh"),
    "watery": ("watery", "aquatic", "fresh", "clean", "transparent"),
    "aquatic": ("aquatic", "marine", "watery", "fresh", "blue"),
    "marine": ("marine", "aquatic", "sea", "oceanic", "salty"),
    "ozonic": ("ozonic", "airy", "fresh", "clean", "watery"),
    "salty": ("salty", "marine", "aquatic", "beachy", "mineral"),
    "green": ("green", "herbal", "leafy", "fresh", "natural"),
    "herbal": ("herbal", "green", "aromatic", "fresh", "sharp"),
    "aromatic": ("aromatic", "herbal", "fresh", "lavender", "clean"),
    "spicy": ("spicy", "warm spicy", "peppery", "bold", "hot"),
    "pepper": ("pepper", "spicy", "sharp", "dry", "warm"),
    "cardamom": ("cardamom", "spicy", "aromatic", "fresh spicy", "cool spice"),
    "cinnamon": ("cinnamon", "spicy", "sweet spice", "warm", "gourmand"),
    "clove": ("clove", "spicy", "warm", "dark spice", "rich"),
    "ginger": ("ginger", "spicy", "fresh spicy", "zesty", "bright"),
    "saffron": ("saffron", "spicy", "leathery", "luxury", "warm"),
    "boozy": ("boozy", "rum", "cognac", "whiskey", "liquor"),
    "rum": ("rum", "boozy", "sweet", "dark", "warm"),
    "cognac": ("cognac", "boozy", "rich", "dark", "warm"),
    "whiskey": ("whiskey", "boozy", "smoky", "dark", "warm"),
}

VIBE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "fresh": ("fresh", "clean", "crisp", "airy", "light"),
    "dark": ("dark", "smoky", "mysterious", "intense", "rich"),
    "sexy": ("sexy", "sensual", "seductive", "date", "night"),
    "clean": ("clean", "soapy", "fresh", "laundry", "musky"),
    "luxury": ("luxury", "premium", "niche", "designer", "refined"),
    "fruity": ("fruity", "juicy", "sweet", "bright", "tropical"),
    "warm": ("warm", "amber", "vanilla", "spicy", "cozy"),
    "elegant": ("elegant", "refined", "classy", "formal", "smooth"),
    "sweet": ("sweet", "gourmand", "vanilla", "caramel", "cozy"),
}

SEASON_SYNONYMS: dict[str, tuple[str, ...]] = {
    "summer": ("summer", "fresh", "citrus", "aquatic", "marine", "bright", "light"),
    "winter": ("winter", "warm", "amber", "vanilla", "oud", "spicy", "sweet"),
    "spring": ("spring", "floral", "green", "bright", "fresh", "airy"),
    "fall": ("fall", "autumn", "woody", "amber", "spicy", "rich", "smooth"),
    "autumn": ("fall", "autumn", "woody", "amber", "spicy", "rich", "smooth"),
    "rainy": ("rainy", "clean", "green", "watery", "soft", "musky"),
    "monsoon": ("monsoon", "rainy", "green", "watery", "clean", "fresh"),
}

OCCASION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "office": ("office", "clean", "fresh", "professional", "safe", "versatile"),
    "work": ("office", "professional", "clean", "safe", "fresh"),
    "daily": ("daily", "versatile", "easy", "clean", "balanced"),
    "gym": ("gym", "fresh", "clean", "citrus", "light", "aquatic"),
    "date": ("date", "sexy", "warm", "vanilla", "amber", "sweet"),
    "date night": ("date", "date night", "sexy", "warm", "amber", "sweet"),
    "party": ("party", "loud", "sweet", "strong", "projection", "playful"),
    "club": ("club", "party", "sweet", "strong", "loud", "beast mode"),
    "wedding": ("wedding", "elegant", "formal", "luxury", "clean", "refined"),
    "formal": ("formal", "elegant", "refined", "luxury", "classy"),
    "casual": ("casual", "easy", "versatile", "fresh", "soft"),
    "vacation": ("vacation", "tropical", "fresh", "marine", "bright", "summer"),
}

PERFORMANCE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "long lasting": ("long lasting", "lasting", "strong", "durable", "performance"),
    "beast mode": ("beast mode", "strong", "powerful", "loud", "projection"),
    "projection": ("projection", "strong", "loud", "beast mode", "sillage"),
    "compliment": ("compliment", "mass appealing", "crowd pleasing", "attractive", "easy like"),
    "versatile": ("versatile", "daily", "all rounder", "balanced", "easy wear"),
}

LUXURY_INTENT_TERMS: set[str] = {
    "luxury", "luxurious", "premium", "designer", "niche", "expensive",
    "highend", "high-end", "famous", "iconic", "best", "bestselling",
    "bestseller", "top", "elite", "classy", "refined"
}

LOW_TIER_BRANDS: set[str] = {
    "zara", "avon", "oriflame", "celine dion", "jennifer lopez"
}

_SPECIAL_APOS = "’`´"
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]+")


def normalize_text(text: str) -> str:
    s = str(text or "")
    for ch in _SPECIAL_APOS:
        s = s.replace(ch, "'")
    s = s.lower().strip()
    s = _NON_ALNUM_SPACE_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def tokenize_text(text: str) -> list[str]:
    n = normalize_text(text)
    if not n:
        return []
    return [t for t in n.split(" ") if t]


def normalize_token(token: str) -> str:
    toks = tokenize_text(token)
    return toks[0] if toks else ""


def dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        k = normalize_text(item)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(item.strip())
    return out


def _build_lookup(mapping: dict[str, tuple[str, ...]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for key, terms in mapping.items():
        nk = normalize_text(key)
        if nk:
            lookup[nk] = nk
        for t in terms:
            nt = normalize_text(t)
            if nt and nt not in lookup:
                lookup[nt] = nk
    return lookup


_NOTE_LOOKUP = _build_lookup(NOTE_SYNONYMS)
_VIBE_LOOKUP = _build_lookup(VIBE_SYNONYMS)
_SEASON_LOOKUP = _build_lookup(SEASON_SYNONYMS)
_OCCASION_LOOKUP = _build_lookup(OCCASION_SYNONYMS)
_PERFORMANCE_LOOKUP = _build_lookup(PERFORMANCE_SYNONYMS)
_VIP_BRANDS_NORM = tuple(sorted((normalize_text(v) for v in VIP_BRANDS), key=len, reverse=True))
_LOW_TIER_BRANDS_NORM = tuple(sorted((normalize_text(v) for v in LOW_TIER_BRANDS), key=len, reverse=True))


def _all_phrase_terms() -> tuple[str, ...]:
    phrase_pool: set[str] = set()
    for d in (NOTE_SYNONYMS, VIBE_SYNONYMS, SEASON_SYNONYMS, OCCASION_SYNONYMS, PERFORMANCE_SYNONYMS):
        for key, values in d.items():
            k = normalize_text(key)
            if " " in k:
                phrase_pool.add(k)
            for v in values:
                nv = normalize_text(v)
                if " " in nv:
                    phrase_pool.add(nv)
    for t in LUXURY_INTENT_TERMS:
        nt = normalize_text(t)
        if " " in nt:
            phrase_pool.add(nt)
    return tuple(sorted(phrase_pool, key=len, reverse=True))


_PHRASE_TERMS = _all_phrase_terms()


def _find_phrases(normalized: str) -> list[str]:
    if not normalized:
        return []
    padded = f" {normalized} "
    found: list[str] = []
    for ph in _PHRASE_TERMS:
        if f" {ph} " in padded:
            found.append(ph)
    return dedupe_keep_order(found)


def _concept_keys(terms: list[str], lookup: dict[str, str]) -> list[str]:
    keys: list[str] = []
    for t in terms:
        nt = normalize_text(t)
        if nt in lookup:
            keys.append(lookup[nt])
    return dedupe_keep_order(keys)


@dataclass
class ParsedQuery:
    raw: str
    normalized: str
    tokens: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)
    exact_note_terms: list[str] = field(default_factory=list)
    vibe_terms: list[str] = field(default_factory=list)
    season_terms: list[str] = field(default_factory=list)
    occasion_terms: list[str] = field(default_factory=list)
    performance_terms: list[str] = field(default_factory=list)
    luxury_intent: bool = False
    perfume_name_like: bool = False
    resolved_index: int | None = None


def parse_query(query: str, df: pd.DataFrame | None = None) -> ParsedQuery:
    raw = str(query or "")
    normalized = normalize_text(raw)
    tokens = tokenize_text(raw)
    phrases = _find_phrases(normalized)
    terms = dedupe_keep_order(tokens + phrases)

    exact_note_terms = _concept_keys(terms, _NOTE_LOOKUP)
    vibe_terms = _concept_keys(terms, _VIBE_LOOKUP)
    season_terms = _concept_keys(terms, _SEASON_LOOKUP)
    occasion_terms = _concept_keys(terms, _OCCASION_LOOKUP)
    performance_terms = _concept_keys(terms, _PERFORMANCE_LOOKUP)

    luxury_intent = any(normalize_text(t) in {normalize_text(x) for x in LUXURY_INTENT_TERMS} for t in terms)

    resolved_index = resolve_perfume_index(df, raw) if df is not None else None
    vip_brand_mentioned = any(v and v in normalized for v in _VIP_BRANDS_NORM) and len(tokens) >= 2

    # Query looks like product name when it has low "vibe language" signal.
    vibe_signal_count = len(exact_note_terms) + len(vibe_terms) + len(season_terms) + len(occasion_terms)
    title_like = bool(re.search(r"[A-Z]", raw)) and len(tokens) <= 5 and vibe_signal_count <= 1

    perfume_name_like = (resolved_index is not None) or vip_brand_mentioned or title_like

    parsed = ParsedQuery(
        raw=raw,
        normalized=normalized,
        tokens=tokens,
        exact_note_terms=exact_note_terms,
        vibe_terms=vibe_terms,
        season_terms=season_terms,
        occasion_terms=occasion_terms,
        performance_terms=performance_terms,
        luxury_intent=luxury_intent,
        perfume_name_like=perfume_name_like,
        resolved_index=resolved_index,
    )
    parsed.expanded_terms = expand_query_terms(parsed)
    return parsed


def expand_query_terms(parsed: ParsedQuery) -> list[str]:
    expanded: list[str] = []
    expanded.extend(parsed.tokens)

    if parsed.perfume_name_like and parsed.resolved_index is not None:
        # Preserve nearest-neighbor behavior for known perfume names.
        return dedupe_keep_order(expanded)

    for nk in parsed.exact_note_terms:
        expanded.extend([nk, nk])
        expanded.extend(NOTE_SYNONYMS.get(nk, ())[:5])

    for vk in parsed.vibe_terms:
        expanded.extend(VIBE_SYNONYMS.get(vk, ())[:4])

    for sk in parsed.season_terms:
        expanded.extend(SEASON_SYNONYMS.get(sk, ())[:4])

    for ok in parsed.occasion_terms:
        expanded.extend(OCCASION_SYNONYMS.get(ok, ())[:4])

    for pk in parsed.performance_terms:
        expanded.extend(PERFORMANCE_SYNONYMS.get(pk, ())[:4])

    if parsed.luxury_intent:
        expanded.extend(["luxury", "premium", "designer", "niche", "expensive", "refined"])

    expanded = dedupe_keep_order(expanded)
    return expanded[:140]


def build_query_text(parsed: ParsedQuery) -> str:
    if parsed.perfume_name_like and parsed.resolved_index is not None:
        return parsed.normalized

    weighted: list[str] = [parsed.normalized]
    for nk in parsed.exact_note_terms:
        weighted.extend([nk, nk])
        weighted.extend(list(NOTE_SYNONYMS.get(nk, ())[:5]))
    weighted.extend(parsed.expanded_terms)
    tokens = tokenize_text(" ".join(weighted))
    return " ".join(tokens[:220])


def _safe(df: pd.DataFrame, idx: int, col: str) -> str:
    if col not in df.columns:
        return ""
    return normalize_text(df.at[idx, col])


def row_note_blob(df: pd.DataFrame, idx: int) -> str:
    if "_notes_blob" in df.columns:
        return _safe(df, idx, "_notes_blob")
    return normalize_text(
        " ".join(
            str(df.at[idx, c]) if c in df.columns else ""
            for c in ("Top Notes", "Middle Notes", "Base Notes")
        )
    )


def row_accord_blob(df: pd.DataFrame, idx: int) -> str:
    if "_accords_blob" in df.columns:
        return _safe(df, idx, "_accords_blob")
    return _safe(df, idx, "Main Accords")


def row_name_blob(df: pd.DataFrame, idx: int) -> str:
    if "_name_blob" in df.columns:
        return _safe(df, idx, "_name_blob")
    return _safe(df, idx, "Perfume Name")


def row_season_blob(df: pd.DataFrame, idx: int) -> str:
    if "_season_blob" in df.columns:
        return _safe(df, idx, "_season_blob")
    return _safe(df, idx, "Season")


def row_brand_blob(df: pd.DataFrame, idx: int) -> str:
    if "_brand_blob" in df.columns:
        return _safe(df, idx, "_brand_blob")
    return _safe(df, idx, "Brand")


def row_full_blob(df: pd.DataFrame, idx: int) -> str:
    if "_full_blob" in df.columns:
        return _safe(df, idx, "_full_blob")
    return normalize_text(
        " ".join(
            [
                row_name_blob(df, idx),
                row_brand_blob(df, idx),
                row_note_blob(df, idx),
                row_accord_blob(df, idx),
                row_season_blob(df, idx),
            ]
        )
    )


def _contains_term(blob: str, term: str) -> bool:
    t = normalize_text(term)
    if not t or not blob:
        return False
    return f" {t} " in f" {blob} "


def _matches_any(blob: str, terms: tuple[str, ...] | list[str]) -> bool:
    return any(_contains_term(blob, t) for t in terms)


def _is_vip_brand(brand_blob: str) -> bool:
    return any(v and _contains_term(brand_blob, v) for v in _VIP_BRANDS_NORM)


def _is_low_tier_brand(brand_blob: str) -> bool:
    return any(v and _contains_term(brand_blob, v) for v in _LOW_TIER_BRANDS_NORM)


def rerank_candidates(
    candidates: list[dict],
    parsed: ParsedQuery,
    df: pd.DataFrame,
) -> list[dict]:
    reranked: list[dict] = []

    for c in candidates:
        r = dict(c)
        idx = int(r["index"])
        similarity = float(r.get("similarity", 0.0))
        if "similarity" not in r:
            r["similarity"] = similarity

        notes_blob = row_note_blob(df, idx)
        accords_blob = row_accord_blob(df, idx)
        name_blob = row_name_blob(df, idx)
        season_blob = row_season_blob(df, idx)
        brand_blob = row_brand_blob(df, idx)
        full_blob = row_full_blob(df, idx)

        score = similarity
        matched_note_terms: list[str] = []
        matched_vibe_terms: list[str] = []
        matched_season_terms: list[str] = []

        # Exact note matching has highest influence for note queries.
        for nk in parsed.exact_note_terms:
            note_terms = (nk,) + NOTE_SYNONYMS.get(nk, ())
            if _matches_any(notes_blob, note_terms) or _matches_any(accords_blob, note_terms) or _matches_any(name_blob, note_terms):
                score += 0.22
                matched_note_terms.append(nk)
            elif _matches_any(full_blob, note_terms):
                score += 0.08
                matched_note_terms.append(nk)

        # Related vibe intent.
        for vk in parsed.vibe_terms:
            terms = (vk,) + VIBE_SYNONYMS.get(vk, ())
            if _matches_any(full_blob, terms):
                score += 0.055
                matched_vibe_terms.append(vk)

        # Season intent.
        for sk in parsed.season_terms:
            terms = (sk,) + SEASON_SYNONYMS.get(sk, ())
            if _matches_any(season_blob, terms):
                score += 0.09
                matched_season_terms.append(sk)
            elif _matches_any(full_blob, terms):
                score += 0.05
                matched_season_terms.append(sk)

        # Occasion + performance intent.
        for ok in parsed.occasion_terms:
            terms = (ok,) + OCCASION_SYNONYMS.get(ok, ())
            if _matches_any(full_blob, terms):
                score += 0.07
                matched_vibe_terms.append(ok)

        for pk in parsed.performance_terms:
            terms = (pk,) + PERFORMANCE_SYNONYMS.get(pk, ())
            if _matches_any(full_blob, terms):
                score += 0.05
                matched_vibe_terms.append(pk)

        is_vip = _is_vip_brand(brand_blob)
        is_low_tier = _is_low_tier_brand(brand_blob)

        vip_boost = 0.14 if (parsed.luxury_intent and is_vip) else (0.06 if is_vip else 0.0)
        low_tier_penalty = -0.12 if (parsed.luxury_intent and is_low_tier) else (-0.05 if is_low_tier else 0.0)
        score += vip_boost + low_tier_penalty

        # Safety: perfume-name-like queries keep similarity dominant.
        if parsed.perfume_name_like:
            score = similarity + (score - similarity) * 0.40

        r["is_vip"] = is_vip
        r["matched_note_terms"] = dedupe_keep_order(matched_note_terms)
        r["matched_vibe_terms"] = dedupe_keep_order(matched_vibe_terms)
        r["matched_season_terms"] = dedupe_keep_order(matched_season_terms)
        r["luxury_boost_applied"] = vip_boost + low_tier_penalty
        r["final_score"] = float(score)
        reranked.append(r)

    reranked.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
    return reranked


def build_metadata(df: pd.DataFrame) -> pd.Series:
    """
    Concatenate fields for TF-IDF. Notes and accords are repeated so vibe queries
    match ingredient language more strongly than name/brand tokens.
    """
    name = df["Perfume Name"].fillna("").astype(str).str.strip()
    brand = df["Brand"].fillna("").astype(str).str.strip()
    top = df["Top Notes"].fillna("").astype(str).str.strip()
    middle = df["Middle Notes"].fillna("").astype(str).str.strip()
    base = df["Base Notes"].fillna("").astype(str).str.strip()
    accords = df["Main Accords"].fillna("").astype(str).str.strip()
    season = df["Season"].fillna("").astype(str).str.strip()

    # Name/brand once, notes ×3, accords ×3, season ×2.
    acc = (
        name + " " + brand + " "
        + top + " " + top + " " + top + " "
        + middle + " " + middle + " " + middle + " "
        + base + " " + base + " " + base + " "
        + accords + " " + accords + " " + accords + " "
        + season + " " + season
    )
    return acc.str.replace(r"\s+", " ", regex=True).str.strip()


def _rating_count_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == "rating count":
            return c
    return None


def filter_by_rating_count(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """Drop rows with fewer than min_count ratings when a Rating Count column exists."""
    col = _rating_count_column(df)
    if col is None:
        return df
    n_before = len(df)
    raw = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
    raw = raw.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    nums = pd.to_numeric(raw, errors="coerce")
    df = df.loc[nums >= min_count].reset_index(drop=True)
    print(
        f"Rating filter ({col} >= {min_count}): {n_before} -> {len(df)} rows "
        f"({n_before - len(df)} removed)"
    )
    return df


def resolve_perfume_index(df: pd.DataFrame, query: str) -> int | None:
    """Return row position if query refers to a perfume name; else None (vibe search)."""
    q = normalize_text(query)
    if not q:
        return None
    names = df["Perfume Name"].fillna("").astype(str)
    lowered = names.str.lower()

    exact = df.index[lowered == q].tolist()
    if len(exact) == 1:
        return int(exact[0])
    if len(exact) > 1:
        return int(exact[0])

    esc = re.escape(q)
    contains = df.index[lowered.str.contains(esc, regex=True, na=False)].tolist()
    if len(contains) == 1:
        return int(contains[0])
    if len(contains) > 1:
        shortest = min(contains, key=lambda i: len(names.at[i]))
        return int(shortest)

    starts = df.index[lowered.str.startswith(q)].tolist()
    if len(starts) >= 1:
        return int(min(starts, key=lambda i: len(names.at[i])))

    return None


def _gender_ok(gender_value: str, gender_filter: str | None) -> bool:
    if gender_filter is None:
        return True
    g = str(gender_value).lower().strip()
    f = str(gender_filter).strip().lower()
    # Dataset uses: for men | for women | for women and men (unisex)
    if f in ("man", "men", "male", "m"):
        return g in ("for men", "for women and men")
    if f in ("woman", "women", "female", "f", "w"):
        return g in ("for women", "for women and men")
    return True


def load_artifacts(directory: str | Path = ".") -> None:
    """Load KNN, vectorizer, and dataframe (e.g. in Flask before handling requests)."""
    global PERFUME_DF, NN_MODEL, TFIDF_VECTORIZER
    d = Path(directory)
    PERFUME_DF = joblib.load(d / ARTIFACT_DF)
    TFIDF_VECTORIZER = joblib.load(d / ARTIFACT_TFIDF)
    NN_MODEL = joblib.load(d / ARTIFACT_KNN)


def _add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_name_blob"] = out["Perfume Name"].fillna("").astype(str).map(normalize_text)
    out["_brand_blob"] = out["Brand"].fillna("").astype(str).map(normalize_text)
    out["_notes_blob"] = (
        out["Top Notes"].fillna("").astype(str)
        + " "
        + out["Middle Notes"].fillna("").astype(str)
        + " "
        + out["Base Notes"].fillna("").astype(str)
    ).map(normalize_text)
    out["_accords_blob"] = out["Main Accords"].fillna("").astype(str).map(normalize_text)
    out["_season_blob"] = out["Season"].fillna("").astype(str).map(normalize_text)
    out["_full_blob"] = (
        out["_name_blob"] + " "
        + out["_brand_blob"] + " "
        + out["_notes_blob"] + " "
        + out["_accords_blob"] + " "
        + out["_season_blob"]
    ).map(normalize_text)
    return out


def get_recommendations(
    query: str,
    gender_filter: str | None = None,
    k: int = 5,
    *,
    df: pd.DataFrame | None = None,
    neighbor_model: NearestNeighbors | None = None,
    vectorizer: TfidfVectorizer | None = None,
) -> list[dict]:
    """
    Similarity: if `query` matches a perfume name, return neighbors of that perfume.
    Vibe: otherwise treat `query` as intent-rich text and rerank with semantic rules.
    """
    df = df if df is not None else PERFUME_DF
    neighbor_model = neighbor_model if neighbor_model is not None else NN_MODEL
    vectorizer = vectorizer if vectorizer is not None else TFIDF_VECTORIZER

    if df is None or neighbor_model is None or vectorizer is None:
        raise RuntimeError("Models not loaded. Run training or load_artifacts().")

    query = (query or "").strip()
    if not query:
        return []

    parsed = parse_query(query, df)
    idx = parsed.resolved_index
    meta_col = "metadata"

    if idx is not None:
        query_text = str(df.at[idx, meta_col])
        exclude_self = idx
    else:
        query_text = build_query_text(parsed)
        exclude_self = None

    xq = vectorizer.transform([query_text])

    base_pool = 300
    if parsed.luxury_intent or (len(parsed.exact_note_terms) == 1 and len(parsed.tokens) <= 3):
        base_pool = 400
    pool = min(base_pool, len(df))
    ranked: list[dict] = []

    while pool <= len(df):
        nv = int(min(pool, len(df)))
        distances, indices = neighbor_model.kneighbors(xq, n_neighbors=nv)
        dist_row = distances[0]
        idx_row = indices[0]

        candidates: list[dict] = []
        for dist, nei in zip(dist_row, idx_row):
            nei = int(nei)
            if exclude_self is not None and nei == exclude_self:
                continue
            if not _gender_ok(df.at[nei, "Gender"], gender_filter):
                continue

            similarity = 1.0 - float(dist)
            candidates.append(
                {
                    "index": nei,
                    "perfume_name": df.at[nei, "Perfume Name"],
                    "brand": df.at[nei, "Brand"],
                    "gender": df.at[nei, "Gender"],
                    "season": df.at[nei, "Season"],
                    "cosine_distance": float(dist),
                    "similarity": float(similarity),
                }
            )

        ranked = sorted(candidates, key=lambda r: r["similarity"], reverse=True)

        if len(ranked) >= k or nv >= len(df):
            results = rerank_candidates(ranked, parsed, df)
            results = results[:k]
            print("Top brands after rerank:", [r.get("brand") for r in results[:5]])
            return results

        pool = min(len(df), max(pool * 2, pool + 100))

    results = rerank_candidates(ranked, parsed, df)
    results = results[:k]
    print("Top brands after rerank:", [r.get("brand") for r in results[:5]])
    return results


def train_and_save(base_dir: str | Path | None = None) -> None:
    global PERFUME_DF, NN_MODEL, TFIDF_VECTORIZER

    base = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    csv_path = base / DATA_CSV
    df = pd.read_csv(csv_path).reset_index(drop=True)
    df = filter_by_rating_count(df, min_count=5)
    df = _add_helper_columns(df)
    df["metadata"] = build_metadata(df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["metadata"])

    nn = NearestNeighbors(n_neighbors=min(50, len(df)), metric="cosine", algorithm="brute")
    nn.fit(X)

    joblib.dump(df, base / ARTIFACT_DF)
    joblib.dump(vectorizer, base / ARTIFACT_TFIDF)
    joblib.dump(nn, base / ARTIFACT_KNN)

    PERFUME_DF = df
    TFIDF_VECTORIZER = vectorizer
    NN_MODEL = nn

    print(f"Saved: {base / ARTIFACT_DF}")
    print(f"Saved: {base / ARTIFACT_TFIDF}")
    print(f"Saved: {base / ARTIFACT_KNN}")
    print(f"Documents: {len(df)}, matrix shape: {X.shape}")


if __name__ == "__main__":
    train_and_save()

    def _print_recs(title: str, recs: list[dict]) -> None:
        print(f"\n--- {title} ---")
        for i, r in enumerate(recs, 1):
            print(
                f"{i}. {r.get('perfume_name')} | {r.get('brand')} | "
                f"sim={float(r.get('similarity', 0.0)):.4f} | "
                f"final={float(r.get('final_score', r.get('similarity', 0.0))):.4f} | "
                f"notes={r.get('matched_note_terms', [])} | "
                f"vibes={r.get('matched_vibe_terms', [])} | "
                f"season={r.get('season')} | vip={r.get('is_vip')}"
            )

    test_queries = [
        "mango",
        "luxury mango summer",
        "fresh clean office",
        "dark oud winter",
        "rose vanilla women",
        "marine summer",
        "beast mode clubbing men",
        "elegant wedding scent",
        "creed aventus",
        "amouage reflection",
    ]

    for q in test_queries:
        _print_recs(f"Query: {q}", get_recommendations(q, k=5))
