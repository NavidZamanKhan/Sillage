"""
Train TF-IDF + NearestNeighbors (cosine) for perfume similarity and vibe search.
"""

from __future__ import annotations

import re
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


def build_metadata(df: pd.DataFrame) -> pd.Series:
    cols = [
        "Perfume Name",
        "Brand",
        "Top Notes",
        "Middle Notes",
        "Base Notes",
        "Main Accords",
        "Season",
    ]
    acc = df[cols[0]].fillna("").astype(str).str.strip()
    for c in cols[1:]:
        acc = acc + " " + df[c].fillna("").astype(str).str.strip()
    return acc.str.replace(r"\s+", " ", regex=True).str.strip()


def resolve_perfume_index(df: pd.DataFrame, query: str) -> int | None:
    """Return row position if query refers to a perfume name; else None (vibe search)."""
    q = query.strip().lower()
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
    Vibe: otherwise treat `query` as text and find similar metadata.

    `gender_filter`: e.g. 'Man' / 'men' -> only 'for men' and unisex ('for women and men').
    """
    df = df if df is not None else PERFUME_DF
    neighbor_model = neighbor_model if neighbor_model is not None else NN_MODEL
    vectorizer = vectorizer if vectorizer is not None else TFIDF_VECTORIZER

    if df is None or neighbor_model is None or vectorizer is None:
        raise RuntimeError("Models not loaded. Run training or load_artifacts().")

    query = (query or "").strip()
    if not query:
        return []

    idx = resolve_perfume_index(df, query)
    meta_col = "metadata"
    if idx is not None:
        text = df.at[idx, meta_col]
        exclude_self = idx
    else:
        text = query
        exclude_self = None

    xq = vectorizer.transform([text])

    results: list[dict] = []
    n_fetch = min(max(k * 20, 30), len(df))

    while len(results) < k and n_fetch <= len(df):
        distances, indices = neighbor_model.kneighbors(xq, n_neighbors=n_fetch)
        dist_row = distances[0]
        idx_row = indices[0]

        seen = {r["index"] for r in results}
        for dist, nei in zip(dist_row, idx_row):
            nei = int(nei)
            if exclude_self is not None and nei == exclude_self:
                continue
            if nei in seen:
                continue
            if not _gender_ok(df.at[nei, "Gender"], gender_filter):
                continue
            # Cosine distance in [0, 2]; similarity for display
            sim = 1.0 - float(dist)
            results.append(
                {
                    "index": nei,
                    "perfume_name": df.at[nei, "Perfume Name"],
                    "brand": df.at[nei, "Brand"],
                    "gender": df.at[nei, "Gender"],
                    "season": df.at[nei, "Season"],
                    "cosine_distance": float(dist),
                    "similarity": sim,
                }
            )
            seen.add(nei)
            if len(results) >= k:
                break

        if n_fetch >= len(df):
            break
        n_fetch = min(n_fetch * 2, len(df))

    return results[:k]


def train_and_save(base_dir: str | Path | None = None) -> None:
    global PERFUME_DF, NN_MODEL, TFIDF_VECTORIZER

    base = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    csv_path = base / DATA_CSV
    df = pd.read_csv(csv_path).reset_index(drop=True)
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
                f"{i}. {r['perfume_name']} | {r['brand']} | {r['gender']} | "
                f"sim={r['similarity']:.4f}"
            )

    _print_recs(
        "Test: vibe 'fresh citrusy summer' (Men)",
        get_recommendations("fresh citrusy summer", gender_filter="Man", k=5),
    )

    _print_recs(
        "Similarity test: perfumes like 'Hacivat'",
        get_recommendations("Hacivat", k=5),
    )

    _print_recs(
        "Vibe test: 'Dark Oud Winter' (Men)",
        get_recommendations("Dark Oud Winter", gender_filter="Man", k=5),
    )

    _print_recs(
        "Gender test: vibe 'Sweet Floral' (Women only)",
        get_recommendations("Sweet Floral", gender_filter="Women", k=5),
    )
