# Hybrid Music Recommendation System

A hybrid music recommendation system that combines **content-based filtering** (audio feature similarity) with **ML-predicted popularity scores** to deliver personalized song recommendations using Spotify track data.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Methodology](#methodology)
- [Models and Results](#models-and-results)
- [Hybrid Recommendation Engine](#hybrid-recommendation-engine)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Overview

Traditional music recommendation systems rely on either collaborative filtering (user behavior) or content-based filtering (audio features) alone. This project implements a **hybrid approach** that:

1. **Predicts song popularity** using supervised ML models (Random Forest, XGBoost) trained on audio features.
2. **Finds sonically similar songs** using cosine similarity via K-Nearest Neighbors on audio feature vectors.
3. **Blends both signals** into a weighted hybrid score to produce final recommendations.

The system also supports an **interactive preference-based interface** where users specify their mood, tempo, danceability, era, and style preferences.

---

## Architecture

```
                    +--------------------------+
                    |    Spotify Track Data     |
                    |    (169,907 tracks)       |
                    +-----------+--------------+
                                |
                    +-----------v--------------+
                    |   Data Cleaning & EDA     |
                    |   (Notebook 1)            |
                    |   - Remove duplicates     |
                    |   - Parse dates           |
                    |   - Filter popularity >30 |
                    |   - MinMaxScaler           |
                    +-----------+--------------+
                                |
                    +-----------v--------------+
                    |  Cleaned Dataset           |
                    |  (93,185 tracks)           |
                    +-----------+--------------+
                                |
              +-----------------+------------------+
              |                                    |
   +----------v-----------+          +-------------v-----------+
   | Popularity Prediction |          | Content-Based Filtering |
   | (XGBoost / RF)        |          | (Cosine Similarity KNN) |
   | R² = 0.47             |          | 9 audio features        |
   +----------+------------+          +-------------+-----------+
              |                                    |
              +-----------------+------------------+
                                |
                    +-----------v--------------+
                    |   Hybrid Scoring          |
                    |   alpha * ML_score +      |
                    |   (1-alpha) * similarity  |
                    +-----------+--------------+
                                |
                    +-----------v--------------+
                    |   Top-N Recommendations   |
                    +--------------------------+
```

---

## Dataset

**Source:** Spotify track dataset

| Property | Value |
|---|---|
| **Raw records** | 169,907 tracks |
| **Cleaned records** | 93,185 tracks |
| **Features** | 19 columns |
| **Time span** | 1921 - 2020 |
| **File (raw)** | `data.csv` (~27 MB) |
| **File (cleaned)** | `cleaned_spotify_data.csv` (~21 MB) |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `id` | string | Spotify track ID |
| `name` | string | Track name |
| `artists` | string | Artist name(s) |
| `duration_ms` | int | Track duration in milliseconds |
| `release_date` | date | Release date |
| `year` | int | Release year |
| `acousticness` | float (0-1) | Confidence the track is acoustic |
| `danceability` | float (0-1) | How suitable for dancing |
| `energy` | float (0-1) | Intensity and activity measure |
| `instrumentalness` | float (0-1) | Predicts if a track has no vocals |
| `liveness` | float (0-1) | Presence of a live audience |
| `loudness` | float (dB) | Overall loudness (normalized to 0-1 after cleaning) |
| `speechiness` | float (0-1) | Presence of spoken words |
| `tempo` | float (BPM) | Estimated tempo (normalized to 0-1 after cleaning) |
| `valence` | float (0-1) | Musical positiveness (happy vs sad) |
| `mode` | int (0/1) | Major (1) or Minor (0) |
| `key` | int (0-11) | Musical key (C=0 through B=11) |
| `popularity` | int (0-100) | Spotify popularity score (target variable) |
| `explicit` | int (0/1) | Whether the track has explicit content |

---

## Project Structure

```
Group13_ML_HybridMusicRecommendationSystem/
|
|-- 01_Data_Exploration_and_Cleaning.ipynb   # EDA and data preprocessing
|-- 02_Modelling_and_Recommendation.ipynb    # Model training and recommendation engine
|-- Final.ipynb                              # Complete end-to-end pipeline with user interaction
|
|-- data.csv                                 # Raw Spotify dataset (169,907 tracks)
|-- cleaned_spotify_data.csv                 # Cleaned and scaled dataset (93,185 tracks)
|
|-- Group13_ML_ProjecrPresentation.pptx      # Project presentation slides
|-- Group13_ML_ProjectReport.pdf             # Detailed project report
|
|-- venv-xgb/                                # Python virtual environment
```

---

## Notebooks

### Notebook 1: Data Exploration and Cleaning
**File:** `01_Data_Exploration_and_Cleaning.ipynb`

- Loads the raw Spotify dataset (169,907 tracks, 19 features)
- Performs exploratory data analysis: shape, dtypes, descriptive statistics
- Checks for missing values (none found) and duplicates (none found)
- Categorizes features into Audio, Metadata, and User Behavior groups
- Cleans data: parses `release_date` to datetime, removes invalid dates
- Filters tracks with `popularity > 30` to focus on relevant songs
- Normalizes 9 audio features using `MinMaxScaler` to [0, 1] range
- Generates boxplots for outlier detection and histograms with KDE for distribution analysis
- Exports cleaned dataset to `cleaned_spotify_data.csv`

### Notebook 2: Modelling and Recommendation
**File:** `02_Modelling_and_Recommendation.ipynb`

- Loads cleaned dataset and visualizes feature correlations via heatmap
- Applies `IterativeImputer` for any residual missing values
- Splits data 80/20 for train/test
- Trains and tunes **Random Forest** and **XGBoost** regressors with `RandomizedSearchCV`
- Evaluates models with RMSE, MAE, R², confusion matrices, classification reports, and ROC curves
- Visualizes feature importance for both models
- Builds content-based recommender using cosine similarity (KNN with `NearestNeighbors`)
- Implements hybrid recommendation function blending ML scores with content similarity

### Final Notebook
**File:** `Final.ipynb`

- Combined end-to-end pipeline (cleaning through recommendation)
- Adds an **interactive user preference system** that prompts for:
  - **Mood:** happy, chill, sad, hype
  - **Tempo:** slow, medium, fast
  - **Danceability:** yes, no
  - **Era:** 90s, 2000s, 2010s, any
  - **Style:** acoustic, electronic
- Filters the dataset based on preferences, then applies hybrid recommendation

---

## Methodology

### 1. Data Preprocessing
- Removed rows with unparseable dates (169,907 → 93,185 after date parsing and popularity filtering)
- Normalized audio features (`acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`) to [0, 1] range using MinMaxScaler
- Extracted `release_year` from `release_date`

### 2. Popularity Prediction (Supervised Learning)
Two regression models predict the popularity score (0-100) of a track from its audio features:

- **Random Forest Regressor** - tuned via RandomizedSearchCV over `n_estimators`, `max_depth`, `min_samples_split`
- **XGBoost Regressor** - tuned via RandomizedSearchCV over `n_estimators`, `max_depth`, `learning_rate`

A binary classification view (popular vs. not) is also derived by thresholding at the median popularity.

### 3. Content-Based Filtering
- Uses K-Nearest Neighbors with **cosine similarity** on the 9-dimensional audio feature space
- Finds the most sonically similar tracks to a given query song

### 4. Hybrid Scoring
The final recommendation score blends both signals:

```
final_score = alpha * (predicted_popularity / 100) + (1 - alpha) * (1 - cosine_distance)
```

Where `alpha = 0.7` by default, giving 70% weight to predicted popularity and 30% to audio similarity.

---

## Models and Results

### Regression Performance (Test Set)

| Metric | Random Forest | XGBoost |
|---|---|---|
| **RMSE** | 8.50 | 8.48 |
| **MAE** | 6.69 | 6.67 |
| **R²** | 0.472 | 0.474 |

### Classification Performance (Popular vs. Not Popular)

| Metric | Random Forest | XGBoost |
|---|---|---|
| **Accuracy** | 74% | 74% |
| **Precision (macro avg)** | 0.74 | 0.74 |
| **Recall (macro avg)** | 0.74 | 0.74 |
| **F1-Score (macro avg)** | 0.74 | 0.74 |

### Key Findings
- **XGBoost slightly outperforms** Random Forest across all metrics
- **Most important features** for popularity prediction: `release_year`, `loudness`, `acousticness`, `danceability`
- `danceability` and `explicit` show the strongest linear correlations with popularity
- Both models achieve AUC scores visible in ROC curve analysis, indicating good discriminative ability

---

## Hybrid Recommendation Engine

### How It Works

1. **Input:** A song (by index or user preference filters)
2. **Content Step:** Find K nearest neighbors by cosine similarity on audio features
3. **ML Step:** Use XGBoost-predicted popularity for each candidate
4. **Blend:** Compute hybrid score with configurable `alpha` weighting
5. **Output:** Top-N recommended songs sorted by hybrid score

### Example Output

```
Input Song Index: 9340

Recommended Songs:
                name   release_date  popularity  predicted_popularity
  Miss Movin' On       2013-07-16      67          55.92
  ULTRAnumb            2011-03-02      57          52.42
  Caboose              1997-01-01      39          43.89
  Electric Eye         1982-01-01      47          41.16
  Helter Skelter       1983-01-01      51          40.96
```

---

## Installation and Setup

### Prerequisites
- Python 3.12+

### Steps

1. **Clone or download** the project folder

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv-xgb
   source venv-xgb/bin/activate        # macOS/Linux
   # venv-xgb\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

5. **Run notebooks in order:**
   - `01_Data_Exploration_and_Cleaning.ipynb` (generates `cleaned_spotify_data.csv`)
   - `02_Modelling_and_Recommendation.ipynb` (trains models and runs recommendations)
   - Or run `Final.ipynb` for the complete pipeline with interactive preference input

---

## Usage

### Get recommendations by song index:
```python
hybrid_recommend(song_index=9340, n=5, alpha=0.7)
```

### Get recommendations by user preferences (Final.ipynb):
```python
recommend_by_preference()
# Prompts for: mood, tempo, danceability, era, style
```

### Adjust the hybrid weighting:
- `alpha = 1.0` → Pure ML-predicted popularity (ignores audio similarity)
- `alpha = 0.0` → Pure content-based filtering (ignores popularity)
- `alpha = 0.7` → Default: 70% popularity, 30% audio similarity

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | >= 2.0 | Data manipulation |
| `numpy` | >= 1.19 | Numerical computation |
| `scikit-learn` | >= 1.6 | ML models, preprocessing, evaluation |
| `xgboost` | >= 2.0 | Gradient boosted tree model |
| `matplotlib` | >= 3.5 | Plotting and visualization |
| `seaborn` | >= 0.12 | Statistical visualization |
| `jupyter` | >= 1.0 | Notebook environment |

---

## License

This project was developed as part of an academic Machine Learning course (Group 13).
