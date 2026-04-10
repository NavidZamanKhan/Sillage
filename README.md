# 🌬️ Sillage | AI-Powered Fragrance Finder

**Sillage** is a machine learning-based web application designed to help users discover perfumes based on "vibes," specific scent notes, or similarity to famous fragrances. 

Built for the **Artificial Intelligence Lab** at Metropolitan University, this project uses a dataset of over 46,000 perfumes and a K-Nearest Neighbors (KNN) algorithm to provide instant, personalized recommendations.

---

## 🚀 Features

* **Instant Vibe Search:** Describe a mood (e.g., "dark oud winter") or an occasion ("fresh gym scent"), and the AI finds the best matches.
* **Similarity Engine:** Search for a specific perfume (e.g., *Nishane Hacivat*) to find scents with a similar DNA.
* **Intelligent Filtering:** Toggle between Men, Women, and Both categories with real-time result updates.
* **Dynamic Loading:** A "Load 5 More" feature allows users to explore deeper into the recommendation list without page refreshes.
* **Direct Discovery:** Every result links directly to a Google search for reviews, prices, and shopping.

---

## 🧠 The "Brain": Machine Learning Logic

The recommendation engine is built on a custom-weighted NLP (Natural Language Processing) pipeline:

1.  **Vectorization:** We use `TfidfVectorizer` to convert perfume metadata (Notes, Accords, Brands, and Seasons) into numerical coordinates.
2.  **Feature Weighting:** To improve accuracy, **Scent Notes are weighted 3x higher** and **Main Accords 2x higher** than the brand name. This ensures the AI prioritizes how a perfume *smells* over who made it.
3.  **Algorithm:** We utilize the **K-Nearest Neighbors (KNN)** algorithm with **Cosine Similarity** to calculate the mathematical distance between your search query and 46,000+ potential matches.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn (KNN, TF-IDF), Pandas, Joblib
* **Frontend:** HTML5, Tailwind CSS, JavaScript (Fetch API)
* **Development Tool:** Cursor AI
* **Dataset:** Cleaned Fragrantica Global Dataset (~46,000 entries)

---

## 📁 Project Structure

```text
fragrance_finder/
├── app.py                 # Flask server & search routes
├── train_model.py         # ML training script & logic
├── prepare_data.py        # Data cleaning & preprocessing
├── fra_perfumes.csv       # Raw dataset
├── cleaned_perfumes.csv   # Pre-processed, AI-ready data
├── templates/
│   └── index.html         # Modern dark-themed UI
└── *.joblib               # Saved ML model artifacts
```

---

## Setup

### 1. Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- Your raw dataset: `fra_perfumes.csv` in the project folder (or use the one already included)

### 2. Virtual environment (recommended)

```bash
cd fragrance_finder
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install flask pandas scikit-learn joblib
```

### 4. Prepare the dataset

Builds `cleaned_perfumes.csv` from `fra_perfumes.csv` (column detection, note parsing, season heuristics, etc.):

```bash
python prepare_data.py
```

### 5. Train the model

Fits TF-IDF + KNN and writes the artifacts the Flask app loads:

- `perfumes_df.joblib`
- `perfumes_tfidf.joblib`
- `perfumes_knn.joblib`

```bash
python train_model.py
```

**Optional quality filter:** If `cleaned_perfumes.csv` includes a **`Rating Count`** column, training automatically drops perfumes with fewer than **5** ratings. To use it, keep or merge that column when you build your cleaned CSV.

### 6. Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000/** in your browser.

Alternatively:

```bash
flask --app app run
```

---

## Updating after data or code changes

1. Re-run **`python prepare_data.py`** if you changed the raw CSV or cleaning logic.
2. Re-run **`python train_model.py`** if you changed `cleaned_perfumes.csv`, `build_metadata`, or vectorizer/KNN settings.
3. **Restart** the Flask process so it reloads the new `.joblib` files.