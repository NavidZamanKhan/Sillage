# Sillage – AI-Based Perfume Recommendation System

> *The scent trail continues.*

An AI-powered web app that recommends perfumes from natural language queries such as moods, notes, seasons, or occasions.

---

## Sillage 2.0 Architecture

Sillage 2.0 splits the original Flask monolith into a modern **Django + Next.js** stack:

```
Sillage/
├── backend/          # Django + Django REST Framework API
│   ├── config/       # Django settings, URLs, WSGI/ASGI
│   ├── recommendations/  # API app (views, serializers, services)
│   └── ml/           # ML engine (unchanged) + data artifacts
├── frontend/         # Next.js (App Router, TypeScript, Tailwind)
│   ├── src/app/      # Pages and global styles
│   ├── src/components/   # React components
│   └── src/lib/      # API client
├── legacy_flask/     # Backup of original Flask app
├── app.py            # Original Flask app (still works)
├── train_model.py    # ML engine (untouched)
└── README.md
```

### Quick Start (Sillage 2.0)

**Backend:**

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
python manage.py runserver
```

**Frontend** (in a separate terminal):

```bash
cd frontend
npm install
npm run dev
```

Then open:
- **Frontend:** [http://localhost:3000](http://localhost:3000)
- **API directly:** [http://127.0.0.1:8000/api/search/?query=rose&gender=Man&limit=5](http://127.0.0.1:8000/api/search/?query=rose&gender=Man&limit=5)

### API Endpoints

```
GET  /api/search/?query=rose&gender=Man&limit=5
POST /api/search/   { "query": "rose", "gender": "Man", "limit": 5 }
```

---

## Overview

Sillage helps users discover fragrances without needing to know brand names or technical perfume vocabulary. A user can type something like _"fresh clean office scent"_ or _"dark oud for winter"_, and the app returns a ranked list of perfumes that match the description.

Under the hood, the app converts perfume metadata (notes, accords, brand, season) into TF-IDF vectors and retrieves the closest matches using K-Nearest Neighbors with cosine similarity. A reranking step then boosts results based on matched intent (notes, vibe, season, occasion) and brand tier.

---

## Features

- **Natural language search** – query by mood, notes, season, or occasion (e.g. _"marine summer"_, _"elegant wedding scent"_).
- **Real-time recommendations** – results are served via a JSON API and rendered instantly on the page.
- **Gender filtering** – filter results for men, women, or unisex.
- **VIP brand prioritization** – niche and premium houses (Amouage, Creed, MFK, Parfums de Marly, Xerjoff, Tom Ford, Dior, Chanel, etc.) are boosted, especially on luxury-intent queries; low-tier brands are penalized.
- **Perfume-name similarity** – typing an actual perfume name (e.g. _"creed aventus"_) returns its nearest neighbors by scent DNA.
- **Weighted feature engineering** – notes and accords dominate the vector representation over names and brands.
- **Clean UI** – glassmorphism Tailwind interface with gender-themed ambient glow.

---

## Tech Stack

**Sillage 2.0:**
- **Backend:** Django + Django REST Framework
- **Frontend:** Next.js (App Router, TypeScript, Tailwind CSS)
- **Machine Learning:** scikit-learn (`TfidfVectorizer`, `NearestNeighbors` with cosine metric)
- **Data:** pandas, NumPy
- **Persistence:** joblib (model artifacts)

**Legacy (v1):**
- **Backend:** Flask
- **Frontend:** HTML, Tailwind CSS (via CDN), vanilla JavaScript

---

## How It Works

The recommendation pipeline:

```
User Query
   │
   ▼
Text Normalization & Tokenization
   │
   ▼
Semantic Expansion
  (notes, vibes, seasons, occasions, performance synonyms)
   │
   ▼
TF-IDF Vectorization
   │
   ▼
KNN Retrieval (Cosine Similarity)
   │
   ▼
Intent-Aware Reranking
  (+ VIP brand boost, gender filter)
   │
   ▼
Ranked Results
```

If the query matches a known perfume name, the app returns nearest neighbors of that perfume. Otherwise it treats the query as a vibe description and performs full semantic retrieval.

---

## Dataset

- **Source:** Fragrantica dataset from Kaggle.
- **Raw file:** `fra_perfumes.csv` (~46k rows).
- **Cleaned file:** `cleaned_perfumes.csv` – produced by `prepare_data.py`.

Fields used:

- `Perfume Name`
- `Brand`
- `Gender` (for men / for women / for women and men)
- `Top Notes`, `Middle Notes`, `Base Notes`
- `Main Accords`
- `Season` (original or inferred from notes/accords)
- `Time of Day` (original or inferred)
- `Rating Count` (used as a quality filter during training when present)

The cleaning step also parses notes and brand from free-text descriptions when structured columns are missing, and heuristically tags season/time-of-day when they are absent.

---

## Model Details

**TF-IDF (Term Frequency – Inverse Document Frequency)**
Each perfume is described by a single "metadata" string that combines its name, brand, notes, accords, and season. TF-IDF turns these strings into numerical vectors where common English words are down-weighted and distinctive scent terms (e.g. _oud_, _bergamot_, _tonka_) carry more signal.

**KNN with Cosine Similarity**
A `NearestNeighbors` model (cosine metric, brute-force) finds the perfumes whose vectors point in the most similar direction to the query vector. Cosine similarity ignores magnitude and focuses on composition, which fits perfume descriptions well.

**Weighted Feature Engineering**
In `build_metadata`, notes and accords are repeated ×3 and season ×2 before vectorization. This makes the model prioritize _how a perfume smells_ over _what it is called_.

**Intent-Aware Reranking**
After KNN retrieval, `rerank_candidates` adds bonuses when a candidate's notes, accords, season, occasion, or performance signals overlap with the parsed query. VIP niche/premium brands receive an additional boost on luxury-intent queries; known low-tier brands receive a small penalty.

---

## Legacy Flask Setup

The original Flask app still works at the project root:

```bash
pip install flask pandas numpy scikit-learn joblib
python app.py
```

Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

A backup of the original Flask files is also in `legacy_flask/`.

---

## Usage

Open the app and type any of:

- A **mood or vibe** — `fresh clean office`, `dark oud winter`, `beast mode clubbing`
- A **note or accord** — `mango`, `rose vanilla`, `marine summer`
- A **season or occasion** — `elegant wedding scent`, `vacation tropical`
- A **perfume name** — `creed aventus`, `amouage reflection`

Optional controls:

- **Gender filter:** `men`, `women`, or both.
- **Limit:** how many results to return (default 5).

---

## Limitations

- **Text-based similarity only** – the model understands words that appear in the dataset. Abstract or metaphorical queries (e.g. _"smells like nostalgia"_) work only when they overlap with indexed note/accord/season vocabulary.
- **No personalization** – results do not adapt to a specific user's history or preferences.
- **No real perfume images** – the UI links to a Google image search for each result instead of hosting bottle images.
- **Dataset bias** – coverage, season tags, and rating counts reflect the source Fragrantica dump and may be incomplete for newer or niche releases.
- **No smell perception** – the system matches language patterns around scents, not actual olfactory similarity.

---

## Future Improvements

- **Deep learning embeddings** (e.g. sentence-transformer models) for richer semantic matching.
- **User accounts and feedback** – thumbs up/down, saved favorites, and implicit signals to personalize ranking.
- **Learning-to-rank** on top of the retrieval layer using user interactions.
- **Hybrid retrieval** combining dense embeddings with the current TF-IDF signal.
- **Richer UI** – faceted filters for brand, season, accord families, and price tier.
- **Real bottle images** via a licensed image source instead of Google search links.
- **PostgreSQL database** – persistent storage for user data, favorites, and analytics.

---

## Authors

- **Navid Zaman Khan**
