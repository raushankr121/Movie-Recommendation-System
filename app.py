
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MovieFlix",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

PLACEHOLDER_POSTER = (
    "https://via.placeholder.com/500x750/1a1a2e/e5e5e5"
    "?text=No+Poster"
)

st.markdown("""
<style>
/* ---------- Global Theme ---------- */
.stApp {
    background-color: #101014;
    color: white;
}
h1, h2, h3 {
    color: white;
}

/* ---------- Movie Card ---------- */
.movie-card {
    background: #1a1a2e;
    border-radius: 12px;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 20px;
    display: block;
}
.movie-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 28px rgba(229, 9, 20, 0.25);
}

/* Card poster image — simple block img with fixed aspect ratio */
.card-poster {
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    display: block;
    background: #16213e;
}

/* Poster wrapper for detail page — uses padding trick for large view */
.poster-wrap {
    position: relative;
    width: 100%;
    padding-top: 150%;
    overflow: hidden;
    background: #16213e;
    display: block;
}
.poster-wrap img {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
}

/* Card body (title + meta) */
.card-body {
    padding: 10px 12px 14px;
    display: block;
}
.movie-title {
    font-size: 14px;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 4px 0;
    height: 42px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    line-height: 1.5;
}
.movie-genre {
    font-size: 12px;
    color: #8d8d8d;
    margin: 0;
    height: 18px;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    display: block;
}
.movie-score {
    font-size: 11px;
    color: #e50914;
    font-weight: 600;
    margin-top: 4px;
    display: block;
}
.badge-watched {
    display: inline-block;
    background: #14321a;
    color: #4ade80;
    border: 1px solid #22c55e;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-top: 5px;
}
.badge-unwatched {
    display: inline-block;
    background: #2d2010;
    color: #fbbf24;
    border: 1px solid #f59e0b;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-top: 5px;
}

/* ---------- Detail Page ---------- */
.detail-header {
    display: block;
    margin-bottom: 8px;
}
.detail-title {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 4px 0;
    line-height: 1.2;
}
.detail-tagline {
    font-size: 16px;
    color: #8d8d8d;
    font-style: italic;
    margin: 0 0 16px 0;
}
.detail-meta {
    display: block;
    margin-bottom: 12px;
}
.detail-meta-item {
    display: inline-block;
    background: #16213e;
    color: #e5e5e5;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    margin: 0 6px 6px 0;
}
.detail-rating-badge {
    display: inline-block;
    background: #e50914;
    color: #ffffff;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    margin: 0 6px 6px 0;
}
.detail-genres {
    display: block;
    margin-bottom: 16px;
}
.detail-genre-tag {
    display: inline-block;
    border: 1px solid #e50914;
    color: #e50914;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    margin: 0 6px 6px 0;
}
.detail-overview-label {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    margin: 20px 0 8px 0;
}
.detail-overview {
    font-size: 15px;
    color: #cccccc;
    line-height: 1.7;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA & COLLABORATIVE FILTERING SETUP ----------------

@st.cache_resource
def compute_user_similarity(train_df, min_common_ratings=5):
    """
    Calculate mean-centered user similarity (Pearson correlation)
    using only common rated movies and applying a minimum threshold.
    """
    R = train_df.values
    mask = ~np.isnan(R)
    u_means = np.nanmean(R, axis=1)
    g_mean = float(np.nanmean(R))
    u_means_clean = np.nan_to_num(u_means, nan=g_mean)

    # Mean-center observed ratings only (unrated entries remain 0.0)
    R_centered = np.where(mask, R - u_means_clean[:, None], 0.0)

    # Vectorized Pearson similarity on co-rated movies only
    var_u = (R_centered ** 2) @ mask.astype(float).T
    denom = np.sqrt(np.maximum(var_u * var_u.T, 0.0))
    numerator = R_centered @ R_centered.T

    sim = np.zeros_like(numerator)
    valid_denom = denom > 1e-9
    sim[valid_denom] = numerator[valid_denom] / denom[valid_denom]

    # Apply minimum common-ratings threshold
    common_counts = mask.astype(int) @ mask.astype(int).T
    sim[common_counts < min_common_ratings] = 0.0
    np.fill_diagonal(sim, 1.0)

    return pd.DataFrame(sim, index=train_df.index, columns=train_df.index)


@st.cache_resource
def load_data():

    train = pd.read_pickle("train_matrix.pkl")
    try:
        user_sim = pd.read_pickle("user_similarity.pkl")
    except Exception:
        user_sim = compute_user_similarity(train, min_common_ratings=5)

    movie_sim = pd.read_pickle("movie_similarity.pkl")

    movies = pd.read_csv("movies.csv")
    links = pd.read_csv("links.csv")

    movie_data = movies.merge(
        links[["movieId", "tmdbId"]],
        on="movieId",
        how="left"
    )

    movie_data["year"] = pd.to_numeric(
        movie_data["title"].str.extract(r'\((\d{4})\)', expand=False),
        errors="coerce"
    )

    return train, user_sim, movie_sim, movie_data


try:
    train_matrix, similarity_df, movie_similarity_df, movies = load_data()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

# Fast O(1) reverse lookup dictionary from tmdbId to movieId
@st.cache_resource
def build_tmdb_to_movie_map(_movies_df):
    valid_tmdb = _movies_df.dropna(subset=["tmdbId"])
    return dict(zip(valid_tmdb["tmdbId"].astype(float), valid_tmdb["movieId"].astype(int)))

tmdb_to_movie_map = build_tmdb_to_movie_map(movies)

# Precompute user rating statistics for mean-centered Collaborative Filtering
user_means = train_matrix.mean(axis=1)
global_mean = float(train_matrix.stack().mean())


# ---------------- CONTENT-BASED FILTERING MODEL ----------------

@st.cache_resource
def build_content_model(_movies_df):
    """Build a memory-efficient TF-IDF + NearestNeighbors content-based model."""
    genres_clean = _movies_df["genres"].fillna("").replace("(no genres listed)", "")
    genres_clean = genres_clean.astype(str).str.replace("|", " ", regex=False)
    titles_clean = _movies_df["title"].fillna("").astype(str)

    # Combine genres (weighted 2x) and title terms
    content_text = genres_clean + " " + genres_clean + " " + titles_clean

    tfidf = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[\w-]+\b",
        min_df=2
    )
    tfidf_matrix = tfidf.fit_transform(content_text)

    nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_model.fit(tfidf_matrix)

    movie_id_to_idx = pd.Series(_movies_df.index.values, index=_movies_df["movieId"])

    return tfidf_matrix, nn_model, movie_id_to_idx


content_tfidf_matrix, content_nn_model, content_movie_map = build_content_model(movies)


# ---------------- TMDB API SETUP ----------------

API_KEY = st.secrets.get("TMDB_API_KEY", "")


@st.cache_resource
def _get_tmdb_session():
    """Create a requests Session with automatic retries for TMDB API."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MovieFlix/2.0"})
    retries = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


TMDB_SESSION = _get_tmdb_session()

@st.cache_data(ttl=86400)
def get_movie_poster(tmdb_id):

    if pd.isna(tmdb_id) or not API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"

    try:
        response = TMDB_SESSION.get(
            url,
            params={"api_key": API_KEY},
            timeout=3.5
        )

        response.raise_for_status()

        poster_path = response.json().get("poster_path")

        if poster_path:
            return (
                "https://image.tmdb.org/t/p/w500"
                + poster_path
            )

    except requests.RequestException:
        pass

    return None


# ---------------- TMDB MOVIE DETAILS ----------------

@st.cache_data(ttl=86400)
def get_movie_details(tmdb_id):
    """Fetch full movie details from TMDB API."""

    if pd.isna(tmdb_id) or not API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"

    try:
        response = TMDB_SESSION.get(
            url,
            params={"api_key": API_KEY},
            timeout=3.5
        )

        response.raise_for_status()
        data = response.json()

        poster_path = data.get("poster_path")
        poster_url = (
            f"https://image.tmdb.org/t/p/w500{poster_path}"
            if poster_path else None
        )

        return {
            "title": data.get("title", "Unknown"),
            "tagline": data.get("tagline", ""),
            "overview": data.get(
                "overview", "No overview available."
            ),
            "release_date": data.get("release_date", ""),
            "runtime": data.get("runtime"),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count", 0),
            "genres": [
                g["name"] for g in data.get("genres", [])
            ],
            "poster_url": poster_url,
        }

    except requests.RequestException:
        pass

    return None


@st.cache_data(ttl=86400)
def get_enriched_tmdb_data(tmdb_id):
    """
    Fetch comprehensive movie information (details, plot overview, keywords, recommendations, similar)
    in a SINGLE optimized API request using TMDB's append_to_response feature.
    Cached for 24 hours with @st.cache_data.
    """
    if pd.isna(tmdb_id) or not tmdb_id or not API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
    params = {
        "api_key": API_KEY,
        "append_to_response": "keywords,recommendations,similar"
    }

    try:
        response = TMDB_SESSION.get(url, params=params, timeout=4.0)
        if response.status_code == 200:
            data = response.json()
            # Extract keyword tokens
            kw_list = []
            for item in data.get("keywords", {}).get("keywords", []):
                name = item.get("name", "").strip().lower()
                if name:
                    kw_list.append(name)

            # Extract TMDB recommendations with their overviews
            rec_items = {}
            for item in data.get("recommendations", {}).get("results", []):
                rec_items[item.get("id")] = {
                    "overview": item.get("overview", ""),
                    "title": item.get("title", ""),
                    "signal": "recommendation"
                }

            # Extract TMDB similar with their overviews
            sim_items = {}
            for item in data.get("similar", {}).get("results", []):
                sim_items[item.get("id")] = {
                    "overview": item.get("overview", ""),
                    "title": item.get("title", ""),
                    "signal": "similar"
                }

            return {
                "title": data.get("title", ""),
                "overview": data.get("overview", ""),
                "genres": [g.get("name", "") for g in data.get("genres", [])],
                "keywords": kw_list,
                "tmdb_recs": rec_items,
                "tmdb_sim": sim_items
            }
    except Exception:
        pass

    return None


# ---------------- USER & HYBRID RECOMMENDATIONS ----------------

def normalize_scores(scores):
    """
    Min-Max normalize scores to [0.0, 1.0].
    Handles constant scores, zero ranges, and missing values safely.
    """
    if scores is None or len(scores) == 0:
        return pd.Series(dtype=float)

    s = scores.dropna()
    if len(s) == 0:
        return pd.Series(0.5, index=scores.index)

    min_val = float(s.min())
    max_val = float(s.max())
    span = max_val - min_val

    if span > 1e-9:
        normalized = (scores - min_val) / span
    else:
        # Constant scores or zero range: map to neutral mid-score 0.5
        normalized = pd.Series(0.5, index=scores.index)

    return normalized.fillna(0.0)


def predict_user_ratings(user_id, k=20, return_raw=False):
    """
    Predict ratings for unwatched movies for user_id using mean-centered
    User-User Collaborative Filtering (Resnick/GroupLens formula).
    Preserves raw unclipped ratings internally for secondary tie-breaking.
    """
    if user_id not in train_matrix.index:
        empty_s = pd.Series(dtype=float)
        return (empty_s, empty_s) if return_raw else empty_s

    user_ratings = train_matrix.loc[user_id]
    user_avg = user_means.get(user_id, global_mean)
    if pd.isna(user_avg):
        user_avg = global_mean

    # Candidate similar users (excluding self, positive correlation only)
    user_sims = similarity_df.loc[user_id].drop(user_id, errors="ignore")
    positive_sims = user_sims[user_sims > 0]

    unwatched = user_ratings[user_ratings.isna()].index

    if len(positive_sims) == 0:
        raw_series = pd.Series(user_avg, index=unwatched)
        disp_series = raw_series.clip(0.5, 5.0)
        return (disp_series, raw_series) if return_raw else disp_series

    # Top k (default 20) most similar users
    top_neighbors = positive_sims.nlargest(k)
    neighbor_ratings = train_matrix.loc[top_neighbors.index]
    neighbor_means = user_means.loc[top_neighbors.index]

    # Center neighbor ratings by subtracting each neighbor's average rating
    neighbor_deviations = neighbor_ratings.sub(neighbor_means, axis=0)

    # Weighted deviation from neighboring users
    weights = top_neighbors
    numerator = neighbor_deviations.mul(weights, axis=0).sum(axis=0)
    denominator = neighbor_ratings.notna().mul(weights.abs(), axis=0).sum(axis=0)

    # Safe division handling zero division or missing values
    weighted_deviation = numerator / denominator.replace(0, np.nan)

    # Raw unclipped predicted rating = user's average + weighted deviation
    raw_ratings = user_avg + weighted_deviation
    raw_ratings = raw_ratings.fillna(user_avg)
    raw_ratings = raw_ratings[user_ratings.isna()].dropna()

    # Displayed predicted rating clipped to valid MovieLens range [0.5, 5.0]
    displayed_ratings = raw_ratings.clip(0.5, 5.0)

    if return_raw:
        return displayed_ratings, raw_ratings
    return displayed_ratings


def predict_item_scores(user_id, k=10):
    """
    Calculate item-based recommendation scores using the top k (10-20)
    most similar movies liked by the user. Ensures strong variation.
    """
    if user_id not in train_matrix.index:
        return pd.Series(dtype=float)

    user_ratings = train_matrix.loc[user_id]
    user_avg = user_means.get(user_id, global_mean)

    # Liked movies (rated >= 4.0; fallback to >= user_avg, then top rated)
    liked_movies = user_ratings[user_ratings >= 4.0].dropna()
    if len(liked_movies) == 0:
        liked_movies = user_ratings[user_ratings >= user_avg].dropna()
    if len(liked_movies) == 0:
        liked_movies = user_ratings.nlargest(10).dropna()

    unwatched = user_ratings[user_ratings.isna()].index
    if len(liked_movies) == 0:
        return pd.Series(user_avg, index=unwatched)

    liked_ids = liked_movies.index.values
    liked_vals = liked_movies.values

    # Extract similarity submatrix (unwatched x liked)
    sim_sub = movie_similarity_df.loc[unwatched, liked_ids].values

    # Select top k most similar liked items per candidate movie
    k_items = min(k, len(liked_ids))
    top_k_idx = np.argpartition(-sim_sub, k_items - 1, axis=1)[:, :k_items]
    rows = np.arange(len(unwatched))[:, None]
    top_sims = sim_sub[rows, top_k_idx]
    top_rats = liked_vals[top_k_idx]

    # Weighted score calculation
    numerator = np.sum(top_sims * top_rats, axis=1)
    denominator = np.sum(np.abs(top_sims), axis=1)
    item_scores_arr = np.where(denominator > 0, numerator / denominator, user_avg)

    return pd.Series(item_scores_arr, index=unwatched)


def predict_content_scores(user_id):
    """
    Calculate content-based recommendation scores measuring cosine similarity
    in TF-IDF space between candidate movies and the user's liked movies profile.
    """
    if user_id not in train_matrix.index:
        return pd.Series(dtype=float)

    user_ratings = train_matrix.loc[user_id]
    user_avg = user_means.get(user_id, global_mean)

    liked_movies = user_ratings[user_ratings >= 4.0].dropna()
    if len(liked_movies) == 0:
        liked_movies = user_ratings[user_ratings >= user_avg].dropna()
    if len(liked_movies) == 0:
        liked_movies = user_ratings.nlargest(10).dropna()

    unwatched = user_ratings[user_ratings.isna()].index
    if len(liked_movies) == 0:
        return pd.Series(0.0, index=unwatched)

    valid_liked = [m for m in liked_movies.index if m in content_movie_map]
    if len(valid_liked) == 0:
        return pd.Series(0.0, index=unwatched)

    liked_indices = content_movie_map.loc[valid_liked].values
    liked_weights = liked_movies.loc[valid_liked].values

    # User content profile: weighted average of liked movies' TF-IDF vectors
    user_profile = np.asarray((liked_weights @ content_tfidf_matrix[liked_indices]) / liked_weights.sum()).ravel()
    profile_norm = np.linalg.norm(user_profile)
    if profile_norm > 1e-9:
        user_profile_unit = user_profile / profile_norm
    else:
        return pd.Series(0.0, index=unwatched)

    valid_unwatched = [m for m in unwatched if m in content_movie_map]
    if len(valid_unwatched) == 0:
        return pd.Series(0.0, index=unwatched)

    unwatched_indices = content_movie_map.loc[valid_unwatched].values
    cand_mat = content_tfidf_matrix[unwatched_indices]

    scores = (cand_mat @ user_profile_unit).ravel()
    return pd.Series(scores, index=valid_unwatched).reindex(unwatched, fill_value=0.0)


def recommend_user_cf(user_id, n=10, k=20):
    """
    Recommend top N movies using pure User-User Collaborative Filtering.
    Uses raw unclipped scores as a secondary tie-breaker when displayed scores are equal.
    """
    disp_scores, raw_scores = predict_user_ratings(user_id, k=k, return_raw=True)
    if disp_scores.empty:
        return pd.DataFrame()

    cand_df = pd.DataFrame({
        "movieId": disp_scores.index.values,
        "cf_score": disp_scores.values,
        "raw_score": raw_scores.values
    })

    # Sort primarily by cf_score, tie-break secondarily by unclipped raw_score
    cand_df = cand_df.sort_values(
        by=["cf_score", "raw_score"],
        ascending=[False, False]
    )
    top_candidates = cand_df.head(n)

    result = movies[movies["movieId"].isin(top_candidates["movieId"])].copy()
    result = result.merge(top_candidates[["movieId", "cf_score", "raw_score"]], on="movieId")
    return result.sort_values(by=["cf_score", "raw_score"], ascending=[False, False])


def hybrid_recommend(user_id, n=10):
    """
    Generate improved hybrid recommendations combining normalized
    User-User CF, Item-Item CF, and Content-Based Filtering signals.
    Preserves raw unclipped predictions internally for ranking and tie-breaking.
    """
    if user_id not in train_matrix.index:
        return pd.DataFrame()

    user_ratings = train_matrix.loc[user_id]
    unwatched = user_ratings[user_ratings.isna()].index

    # 1. User-User CF component (Resnick mean-centered formula)
    disp_user, raw_user = predict_user_ratings(user_id, k=20, return_raw=True)

    # 2. Item-Item CF component (top 10 most relevant liked movies)
    raw_item = predict_item_scores(user_id, k=10)

    # 3. Content-Based TF-IDF component (cosine similarity to user taste profile)
    raw_content = predict_content_scores(user_id)

    # Normalize each score distribution to [0.0, 1.0] with safe handling of constant ranges
    norm_user = normalize_scores(raw_user.reindex(unwatched))
    norm_item = normalize_scores(raw_item.reindex(unwatched))
    norm_content = normalize_scores(raw_content.reindex(unwatched))

    # Balanced weighting strategy: 50% User-CF, 30% Item-CF, 20% Content-Based
    w_user, w_item, w_content = 0.50, 0.30, 0.20
    hybrid_norm = (
        w_user * norm_user +
        w_item * norm_item +
        w_content * norm_content
    )

    # Map normalized hybrid score [0, 1] to predicted rating scale [0.5, 5.0]
    raw_hybrid = 0.5 + 4.5 * hybrid_norm
    disp_hybrid = raw_hybrid.clip(0.5, 5.0).round(4)

    cand_df = pd.DataFrame({
        "movieId": unwatched.values,
        "hybrid_score": disp_hybrid.values,
        "raw_hybrid_score": raw_hybrid.values,
        "user_cf_score": raw_user.reindex(unwatched).values,
        "item_cf_score": raw_item.reindex(unwatched).values,
        "content_score": raw_content.reindex(unwatched).values,
    })

    # Sort primarily by displayed score, using raw_hybrid_score as exact secondary tie-breaker
    cand_df = cand_df.sort_values(
        by=["hybrid_score", "raw_hybrid_score"],
        ascending=[False, False]
    )
    top_candidates = cand_df.head(n)

    result = movies[movies["movieId"].isin(top_candidates["movieId"])].copy()
    result = result.merge(
        top_candidates[[
            "movieId", "hybrid_score", "raw_hybrid_score",
            "user_cf_score", "item_cf_score", "content_score"
        ]],
        on="movieId"
    )
    return result.sort_values(
        by=["hybrid_score", "raw_hybrid_score"],
        ascending=[False, False]
    )


def evaluate_collaborative_filtering(train_df=None, test_ratio=0.2, random_state=42, min_common_ratings=5, k=20):
    """
    Evaluate User-User Collaborative Filtering, Item-Item CF, and Hybrid CF on an 80/20 train/test split.
    Prevents data leakage by calculating user means and similarities exclusively from training data.
    Returns comparison metrics (RMSE, MAE, score variances) across baseline and improved models.
    """
    if train_df is None:
        train_df = train_matrix

    R = train_df.values
    users_idx, movies_idx = np.where(~np.isnan(R))
    n_ratings = len(users_idx)

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_ratings)
    test_size = int(test_ratio * n_ratings)

    test_u = users_idx[perm[:test_size]]
    test_m = movies_idx[perm[:test_size]]
    R_test_actual = R[test_u, test_m].copy()

    # Mask out test ratings in training matrix
    R_train = R.copy()
    R_train[test_u, test_m] = np.nan
    mask_train = ~np.isnan(R_train)

    # 1. Baseline Model (Raw Cosine on zero-filled ratings, top-k neighbors)
    R_train_zero = np.nan_to_num(R_train, nan=0.0)
    sim_raw = cosine_similarity(R_train_zero)
    np.fill_diagonal(sim_raw, 0.0)

    preds_baseline = []
    for u, m in zip(test_u, test_m):
        sims = sim_raw[u]
        top_k_users = np.argsort(sims)[::-1][:k]
        w = sims[top_k_users]
        rats = R_train_zero[top_k_users, m]
        valid = rats > 0
        if valid.sum() > 0:
            denom = np.sum(np.abs(w[valid]))
            pred = np.sum(w[valid] * rats[valid]) / denom if denom > 0 else np.nan
        else:
            pred = np.nan
        preds_baseline.append(pred)

    preds_baseline = np.array(preds_baseline)
    valid_b = ~np.isnan(preds_baseline)
    rmse_baseline = np.sqrt(np.mean((R_test_actual[valid_b] - preds_baseline[valid_b]) ** 2))
    mae_baseline = np.mean(np.abs(R_test_actual[valid_b] - preds_baseline[valid_b]))

    # 2. Improved User-CF Model (Mean-Centered Pearson on Common Items, min_common threshold, top-k neighbors)
    u_means = np.nanmean(R_train, axis=1)
    g_mean = float(np.nanmean(R_train))
    u_means_clean = np.nan_to_num(u_means, nan=g_mean)

    R_centered = np.where(mask_train, R_train - u_means_clean[:, None], 0.0)
    var_u = (R_centered ** 2) @ mask_train.astype(float).T
    denom_sim = np.sqrt(np.maximum(var_u * var_u.T, 0.0))
    num_sim = R_centered @ R_centered.T

    sim_improved = np.zeros_like(num_sim)
    valid_d = denom_sim > 1e-9
    sim_improved[valid_d] = num_sim[valid_d] / denom_sim[valid_d]

    common_counts = mask_train.astype(int) @ mask_train.astype(int).T
    sim_improved[common_counts < min_common_ratings] = 0.0
    np.fill_diagonal(sim_improved, 0.0)

    # Pre-extract movie similarities for Item-CF test evaluation
    col_movie_ids = train_df.columns.values
    movie_sim_mat = movie_similarity_df.loc[col_movie_ids, col_movie_ids].values

    preds_improved = []
    preds_item_cf = []
    preds_hybrid = []

    for u, m in zip(test_u, test_m):
        u_avg = u_means_clean[u]
        sims = sim_improved[u]
        pos_mask = sims > 0
        if pos_mask.sum() > 0:
            pos_idx = np.where(pos_mask)[0]
            top_neighbors = pos_idx[np.argsort(sims[pos_idx])[::-1][:k]]
            rated_mask = mask_train[top_neighbors, m]
            if rated_mask.sum() > 0:
                k_rated = top_neighbors[rated_mask]
                w = sims[k_rated]
                dev = R_centered[k_rated, m]
                denom = np.sum(np.abs(w))
                p_user = u_avg + (np.sum(w * dev) / denom) if denom > 0 else u_avg
            else:
                p_user = u_avg
        else:
            p_user = u_avg
        preds_improved.append(np.clip(p_user, 0.5, 5.0))

        # Item-CF evaluation
        u_rats = R_train[u]
        u_rated_mask = mask_train[u]
        u_liked = np.where(u_rated_mask & (u_rats >= 4.0))[0]
        if len(u_liked) == 0:
            u_liked = np.where(u_rated_mask & (u_rats >= u_avg))[0]
        if len(u_liked) == 0:
            u_liked = np.where(u_rated_mask)[0]

        if len(u_liked) > 0:
            sims_m = movie_sim_mat[m, u_liked]
            k_item = min(10, len(u_liked))
            top_k_item_idx = np.argsort(sims_m)[::-1][:k_item]
            top_sims = sims_m[top_k_item_idx]
            top_rats = u_rats[u_liked[top_k_item_idx]]
            d_item = np.sum(np.abs(top_sims))
            p_item = np.sum(top_sims * top_rats) / d_item if d_item > 0 else u_avg
        else:
            p_item = u_avg
        preds_item_cf.append(np.clip(p_item, 0.5, 5.0))

        # Hybrid blend
        p_hyb = 0.55 * p_user + 0.45 * p_item
        preds_hybrid.append(np.clip(p_hyb, 0.5, 5.0))

    preds_improved = np.array(preds_improved)
    rmse_improved = np.sqrt(np.mean((R_test_actual - preds_improved) ** 2))
    mae_improved = np.mean(np.abs(R_test_actual - preds_improved))

    preds_item_cf = np.array(preds_item_cf)
    rmse_item_cf = np.sqrt(np.mean((R_test_actual - preds_item_cf) ** 2))
    mae_item_cf = np.mean(np.abs(R_test_actual - preds_item_cf))

    preds_hybrid = np.array(preds_hybrid)
    rmse_hybrid = np.sqrt(np.mean((R_test_actual - preds_hybrid) ** 2))
    mae_hybrid = np.mean(np.abs(R_test_actual - preds_hybrid))

    return {
        "baseline": {"rmse": rmse_baseline, "mae": mae_baseline, "coverage": float(valid_b.mean())},
        "user_cf": {"rmse": rmse_improved, "mae": mae_improved, "coverage": 1.0},
        "item_cf": {"rmse": rmse_item_cf, "mae": mae_item_cf, "coverage": 1.0},
        "hybrid_cf": {"rmse": rmse_hybrid, "mae": mae_hybrid, "coverage": 1.0},
        "rmse_reduction": rmse_baseline - rmse_improved,
        "rmse_improvement_pct": (rmse_baseline - rmse_improved) / rmse_baseline * 100,
        "mae_reduction": mae_baseline - mae_improved,
        "mae_improvement_pct": (mae_baseline - mae_improved) / mae_baseline * 100,
        "test_size": test_size,
        "train_size": len(perm) - test_size
    }



# ---------------- POPULARITY & COLD-START RECOMMENDATIONS ----------------

@st.cache_data
def get_bayesian_popular_movies():
    """
    Precompute Bayesian Weighted Rating for movies in train_matrix (IMDb formula):
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
      v = rating count
      R = average movie rating
      C = global mean rating across all ratings
      m = minimum ratings threshold (50th percentile)
    """
    v = train_matrix.notna().sum(axis=0)
    R = train_matrix.mean(axis=0)
    C = global_mean
    m = float(v.quantile(0.50))
    wr = (v / (v + m)) * R + (m / (v + m)) * C
    df = pd.DataFrame({"movieId": wr.index, "wr_score": wr.values})
    return df.sort_values("wr_score", ascending=False)


bayesian_pop_df = get_bayesian_popular_movies()


def get_popularity_recommendations(n=10, watched_ids=None, genres=None):
    """
    Return top N popular movies using Bayesian Weighted Rating (IMDb formula).
    Filters out watched_ids and optionally prioritizes preferred genres.
    """
    if watched_ids is None:
        watched_ids = set()
    else:
        watched_ids = set(watched_ids)

    cand_ids = [m for m in bayesian_pop_df["movieId"] if m not in watched_ids]
    cand_movies = movies[movies["movieId"].isin(cand_ids)].merge(
        bayesian_pop_df, on="movieId"
    ).sort_values("wr_score", ascending=False)

    if genres:
        genre_set = {g.strip().lower() for g in genres if g.strip()}
        if genre_set:
            def matches_genre(g_str):
                if not isinstance(g_str, str):
                    return False
                movie_g = {g.strip().lower() for g in g_str.split("|")}
                return len(movie_g & genre_set) > 0

            filtered = cand_movies[cand_movies["genres"].apply(matches_genre)]
            if len(filtered) >= n:
                cand_movies = filtered
            elif len(filtered) > 0:
                remaining = cand_movies[~cand_movies["movieId"].isin(filtered["movieId"])]
                cand_movies = pd.concat([filtered, remaining])

    res = cand_movies.head(n).copy()
    res["score"] = res["wr_score"].round(2)
    return res


# ---------------- COLD-START PREFERENCES PERSISTENCE ----------------

USER_PREFERENCES_FILE = "user_preferences.json"


def load_user_preferences():
    """Load cold-start user preferences from user_preferences.json."""
    if not os.path.exists(USER_PREFERENCES_FILE):
        return {"favourite_genres": [], "favourite_movie_ids": [], "user_ratings": {}}
    try:
        with open(USER_PREFERENCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {
                    "favourite_genres": data.get("favourite_genres", []),
                    "favourite_movie_ids": data.get("favourite_movie_ids", []),
                    "user_ratings": {int(k): float(v) for k, v in data.get("user_ratings", {}).items() if str(k).isdigit()}
                }
            return {"favourite_genres": [], "favourite_movie_ids": [], "user_ratings": {}}
    except Exception:
        return {"favourite_genres": [], "favourite_movie_ids": [], "user_ratings": {}}


def save_user_preferences(prefs):
    """Save cold-start user preferences to user_preferences.json."""
    try:
        serializable_prefs = {
            "favourite_genres": prefs.get("favourite_genres", []),
            "favourite_movie_ids": prefs.get("favourite_movie_ids", []),
            "user_ratings": {str(k): float(v) for k, v in prefs.get("user_ratings", {}).items()}
        }
        with open(USER_PREFERENCES_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_prefs, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        return False


def recommend_cold_start_user(favourite_genres=None, favourite_movie_ids=None, user_ratings=None, n=10, transition_threshold=10):
    """
    Generate cold-start recommendations for new users:
    - Combines favourite genres, favourite movies, and Bayesian popularity.
    - If user provides 3-5 ratings, smoothly transitions to collaborative filtering.
    - Excludes already watched/selected movies.
    """
    favourite_genres = favourite_genres or []
    favourite_movie_ids = favourite_movie_ids or []
    user_ratings = user_ratings or {}

    watched_ids = set(favourite_movie_ids) | set(user_ratings.keys())

    # Candidate movies from CF catalog excluding watched
    cf_candidate_ids = set(train_matrix.columns) - watched_ids

    # 1. Content component from favourite movies
    content_scores = pd.Series(0.0, index=list(cf_candidate_ids))
    valid_favs = [m for m in favourite_movie_ids if m in content_movie_map]
    if valid_favs:
        fav_indices = content_movie_map.loc[valid_favs].values
        fav_profile = np.asarray(content_tfidf_matrix[fav_indices].mean(axis=0)).ravel()
        p_norm = np.linalg.norm(fav_profile)
        if p_norm > 1e-9:
            fav_profile_unit = fav_profile / p_norm
            cand_valid = [m for m in cf_candidate_ids if m in content_movie_map]
            cand_indices = content_movie_map.loc[cand_valid].values
            cand_mat = content_tfidf_matrix[cand_indices]
            sims = (cand_mat @ fav_profile_unit).ravel()
            content_scores.loc[cand_valid] = sims

    # 2. Genre preference component
    genre_scores = pd.Series(0.0, index=list(cf_candidate_ids))
    if favourite_genres:
        genre_set = {g.strip().lower() for g in favourite_genres if g.strip()}
        cand_movies_df = movies[movies["movieId"].isin(cf_candidate_ids)]
        def calc_genre_overlap(g_str):
            if not isinstance(g_str, str) or not g_str:
                return 0.0
            mg = {g.strip().lower() for g in g_str.split("|")}
            overlap = len(mg & genre_set)
            return overlap / len(genre_set) if genre_set else 0.0
        g_overlaps = cand_movies_df.set_index("movieId")["genres"].apply(calc_genre_overlap)
        genre_scores.update(g_overlaps)

    # 3. Popularity component (Bayesian WR normalized)
    pop_map = bayesian_pop_df.set_index("movieId")["wr_score"]
    cand_pop = pop_map.reindex(list(cf_candidate_ids)).fillna(global_mean)
    min_pop, max_pop = float(cand_pop.min()), float(cand_pop.max())
    pop_span = max_pop - min_pop
    if pop_span > 1e-9:
        pop_scores = (cand_pop - min_pop) / pop_span
    else:
        pop_scores = pd.Series(0.5, index=list(cf_candidate_ids))

    # Normalize helper
    def norm_s(s):
        s_clean = s.dropna()
        if len(s_clean) == 0:
            return pd.Series(0.0, index=s.index)
        mn, mx = float(s_clean.min()), float(s_clean.max())
        return (s - mn) / (mx - mn) if (mx - mn) > 1e-9 else pd.Series(0.0, index=s.index)

    norm_content = norm_s(content_scores)
    norm_genre = norm_s(genre_scores)
    norm_pop = pop_scores

    has_favs = len(valid_favs) > 0
    has_genres = len(favourite_genres) > 0

    if has_favs and has_genres:
        w_c, w_g, w_p = 0.45, 0.35, 0.20
    elif has_genres:
        w_c, w_g, w_p = 0.00, 0.70, 0.30
    elif has_favs:
        w_c, w_g, w_p = 0.75, 0.00, 0.25
    else:
        w_c, w_g, w_p = 0.00, 0.00, 1.00

    cold_start_score = w_c * norm_content + w_g * norm_genre + w_p * norm_pop

    # 4. Collaborative Filtering component (Item-CF)
    cf_scores = pd.Series(0.0, index=list(cf_candidate_ids))
    num_ratings = len(user_ratings)

    if num_ratings > 0:
        valid_rated = {m: r for m, r in user_ratings.items() if m in movie_similarity_df.columns}
        if valid_rated:
            unwatched_list = list(cf_candidate_ids)
            rated_ids = list(valid_rated.keys())
            rated_vals = np.array([valid_rated[m] for m in rated_ids])
            sim_sub = movie_similarity_df.loc[unwatched_list, rated_ids].values

            numerator = sim_sub @ rated_vals
            denominator = np.sum(np.abs(sim_sub), axis=1)
            cf_vals = np.where(denominator > 0, numerator / denominator, np.mean(rated_vals))
            cf_series = pd.Series(cf_vals, index=unwatched_list)
            cf_scores = norm_s(cf_series)

    # Adaptive transition weight: alpha in [0, 1]
    alpha = min(1.0, float(num_ratings) / float(transition_threshold))
    final_score = alpha * cf_scores + (1.0 - alpha) * cold_start_score

    # Map to [0.5, 5.0] rating scale
    disp_rating = (0.5 + 4.5 * final_score).clip(0.5, 5.0).round(2)

    cand_df = pd.DataFrame({
        "movieId": list(cf_candidate_ids),
        "score": disp_rating.values,
        "raw_score": final_score.values,
        "cold_start_score": cold_start_score.values,
        "cf_score": cf_scores.values,
        "alpha": alpha
    }).sort_values(by=["score", "raw_score"], ascending=[False, False])

    top_cands = cand_df.head(n)
    res = movies[movies["movieId"].isin(top_cands["movieId"])].merge(
        top_cands, on="movieId"
    ).sort_values(by=["score", "raw_score"], ascending=[False, False])
    return res


# ---------------- MOVIE TITLE SEARCH (OPTIMIZED) ----------------

@st.cache_data
def search_movies_by_title(query, limit=50):
    """
    Fast search for movies by title with case-insensitive substring and multi-word token matching.
    Handles partial queries (e.g. 'Krr' or 'Krrish'), multi-word queries ('The Dark Knight'),
    and returns matching movies prioritized by popularity.
    """
    if not query or not str(query).strip():
        return movies.head(0)

    q = str(query).strip().lower()
    titles_series = movies["title"].astype(str)

    # 1. Exact case-insensitive substring match
    mask_substr = titles_series.str.lower().str.contains(q, regex=False, na=False)
    matches = movies[mask_substr]

    # 2. Multi-word token match (e.g., 'The Dark Knight' matches 'Dark Knight, The (2008)')
    tokens = [t for t in re.split(r'\W+', q) if t]
    if len(tokens) > 1 and len(matches) < limit:
        def match_tokens(s):
            s_low = str(s).lower()
            return all(tok in s_low for tok in tokens)

        mask_tokens = titles_series.apply(match_tokens)
        token_matches = movies[mask_tokens & ~mask_substr]
        matches = pd.concat([matches, token_matches])

    # Prioritize popular titles using bayesian_pop_df
    try:
        pop_map = bayesian_pop_df.set_index("movieId")["wr_score"]
        matches = matches.copy()
        matches["_search_pop"] = matches["movieId"].map(pop_map).fillna(0.0)
        matches = matches.sort_values(by=["_search_pop", "movieId"], ascending=[False, True])
    except Exception:
        pass

    return matches.head(limit)


# ---------------- SIMILAR MOVIES (CONTENT-BASED) ----------------

@st.cache_data
def recommend_similar_movies(movie_id, n=10):
    """
    Multi-stage content-based recommendation engine:
    Stage 1 (Candidate Generation):
      - TF-IDF NearestNeighbors candidates (top 40)
      - TMDB Editorial/Community recommendations (up to 20)
      - TMDB Plot/Genre-similar movies (up to 20)
      - Candidates merged and mapped into MovieFlix catalog, excluding self.
    Stage 2 (Multi-Signal Relevance Scoring):
      - Plot & Overview Semantic Similarity (TF-IDF cosine similarity on overviews)
      - Thematic Keyword Matching (TMDB keyword tags)
      - Genre Alignment (Jaccard similarity on pipe-separated genres)
      - TMDB Community/Editorial Signal (bonus for direct recommendation/similarity endorsements)
      - Base TF-IDF Content Score (title terms and structural genres)
    Stage 3 (Ranking & Output):
      - Ranked by combined relevance score.
      - Returns top N distinct catalog movies.
      - Graceful fallback to base TF-IDF model if TMDB is offline or unindexed.
    """

    if movie_id not in content_movie_map:
        pop = get_popularity_recommendations(n=n)
        pop["similarity_score"] = 0.50
        return pop

    source_idx = content_movie_map[movie_id]
    source_row = movies.iloc[source_idx]
    source_tmdb = source_row.get("tmdbId")

    # 1. Base TF-IDF Candidates (top 40)
    k_query = min(40, len(movies))
    dists, inds = content_nn_model.kneighbors(content_tfidf_matrix[source_idx], n_neighbors=k_query)
    dists = dists.flatten()
    inds = inds.flatten()
    mask = inds != source_idx

    tfidf_cands = {}
    for idx_val, dist_val in zip(inds[mask], dists[mask]):
        mid = int(movies.iloc[idx_val]["movieId"])
        tfidf_cands[mid] = max(0.0, 1.0 - dist_val)

    # 2. TMDB Enrichment & Candidate Expansion (single request via append_to_response)
    tmdb_data = get_enriched_tmdb_data(source_tmdb)

    tmdb_cand_signals = {}
    tmdb_cand_overviews = {}

    source_overview = ""
    source_keywords = []
    source_genres = set(source_row["genres"].split("|")) if pd.notna(source_row["genres"]) and source_row["genres"] != "(no genres listed)" else set()

    if tmdb_data:
        source_overview = tmdb_data.get("overview", "")
        source_keywords = tmdb_data.get("keywords", [])
        if tmdb_data.get("genres"):
            source_genres.update(tmdb_data["genres"])

        # Add TMDB recommendations
        for tid, tinfo in tmdb_data.get("tmdb_recs", {}).items():
            if float(tid) in tmdb_to_movie_map:
                mid = int(tmdb_to_movie_map[float(tid)])
                if mid != movie_id:
                    tmdb_cand_signals[mid] = 1.0
                    if tinfo.get("overview"):
                        tmdb_cand_overviews[mid] = tinfo["overview"]

        # Add TMDB similar
        for tid, tinfo in tmdb_data.get("tmdb_sim", {}).items():
            if float(tid) in tmdb_to_movie_map:
                mid = int(tmdb_to_movie_map[float(tid)])
                if mid != movie_id and mid not in tmdb_cand_signals:
                    tmdb_cand_signals[mid] = 0.8
                    if tinfo.get("overview"):
                        tmdb_cand_overviews[mid] = tinfo["overview"]

    # Merge candidates (preserving order, removing self and duplicates)
    all_candidate_ids = list(dict.fromkeys(list(tmdb_cand_signals.keys()) + list(tfidf_cands.keys())))
    if movie_id in all_candidate_ids:
        all_candidate_ids.remove(movie_id)

    if not all_candidate_ids:
        pop = get_popularity_recommendations(n=n)
        pop["similarity_score"] = 0.50
        return pop

    cand_df = movies[movies["movieId"].isin(all_candidate_ids)].copy().set_index("movieId")

    # 3. Multi-Signal Scoring
    # A. Genre similarity (Jaccard)
    genre_scores = {}
    for mid, row in cand_df.iterrows():
        c_genres = set(row["genres"].split("|")) if pd.notna(row["genres"]) and row["genres"] != "(no genres listed)" else set()
        intersection = len(source_genres & c_genres)
        union = len(source_genres | c_genres)
        genre_scores[mid] = intersection / union if union > 0 else 0.0

    # B. Keyword & Theme matching
    keyword_scores = {}
    if source_keywords:
        kw_tokens = set()
        for kw in source_keywords:
            for tok in re.split(r'\W+', kw.lower()):
                if len(tok) >= 3:
                    kw_tokens.add(tok)

        for mid, row in cand_df.iterrows():
            c_text = (str(row["title"]) + " " + str(row["genres"]) + " " + tmdb_cand_overviews.get(mid, "")).lower()
            matches_count = sum(1 for tok in kw_tokens if tok in c_text)
            keyword_scores[mid] = min(1.0, matches_count / max(1, min(len(kw_tokens), 4)))
    else:
        keyword_scores = {mid: 0.0 for mid in all_candidate_ids}

    # C. Plot Overview Similarity (TF-IDF cosine on overviews)
    plot_scores = {}
    if source_overview and len(source_overview.strip()) > 10:
        overviews_corpus = [source_overview]
        cand_ids_with_overview = []
        for mid in all_candidate_ids:
            ov = tmdb_cand_overviews.get(mid, "")
            if ov:
                overviews_corpus.append(ov)
                cand_ids_with_overview.append(mid)

        if len(cand_ids_with_overview) > 0:
            try:
                ov_vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
                ov_tfidf = ov_vectorizer.fit_transform(overviews_corpus)
                cos_sims = cosine_similarity(ov_tfidf[0:1], ov_tfidf[1:]).flatten()
                for c_mid, sim_val in zip(cand_ids_with_overview, cos_sims):
                    plot_scores[c_mid] = float(sim_val)
            except Exception:
                pass

    for mid in all_candidate_ids:
        if mid not in plot_scores:
            plot_scores[mid] = 0.0

    # D. Combined Relevance Score
    final_scores = {}
    for mid in all_candidate_ids:
        s_tmdb = tmdb_cand_signals.get(mid, 0.0)
        s_plot = plot_scores.get(mid, 0.0)
        s_kw = keyword_scores.get(mid, 0.0)
        s_genre = genre_scores.get(mid, 0.0)
        s_tfidf = tfidf_cands.get(mid, 0.0)

        if tmdb_data:
            score = (
                0.35 * s_tmdb +
                0.25 * s_plot +
                0.20 * s_kw +
                0.10 * s_genre +
                0.10 * s_tfidf
            )
        else:
            score = 0.60 * s_tfidf + 0.40 * s_genre

        final_scores[mid] = round(score, 4)

    cand_df["similarity_score"] = pd.Series(final_scores)
    cand_df = cand_df.sort_values(by="similarity_score", ascending=False).reset_index()
    return cand_df.head(n)


# ---------------- WATCHLIST PERSISTENCE ----------------

WATCHLIST_FILE = "watchlist.json"


def load_watchlist():
    """Load saved movies from watchlist.json safely."""
    if not os.path.exists(WATCHLIST_FILE):
        return []
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def save_watchlist(watchlist):
    """Save watchlist to watchlist.json safely."""
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(watchlist, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving watchlist: {e}")


def is_in_watchlist(movie_id):
    """Check if a movie_id is in the current watchlist."""
    watchlist = st.session_state.get("watchlist", [])
    try:
        target_id = int(movie_id)
        return any(int(item.get("movieId", -1)) == target_id for item in watchlist)
    except (ValueError, TypeError):
        return False


def add_to_watchlist(movie):
    """Add a movie to the watchlist, preventing duplicates."""
    try:
        m_id = int(movie["movieId"])
    except Exception:
        return False, "Invalid movie ID."

    if is_in_watchlist(m_id):
        return False, f"'{movie.get('title', 'Movie')}' is already in your watchlist."

    tmdb_id = movie.get("tmdbId")
    tmdb_val = float(tmdb_id) if pd.notna(tmdb_id) else None
    title = str(movie.get("title", "Unknown"))

    item = {
        "movieId": m_id,
        "title": title,
        "genres": str(movie.get("genres", "")),
        "tmdbId": tmdb_val,
        "watched": False
    }

    watchlist = load_watchlist()
    if any(int(x.get("movieId", -1)) == m_id for x in watchlist):
        st.session_state["watchlist"] = watchlist
        return False, f"'{title}' is already in your watchlist."

    watchlist.append(item)
    save_watchlist(watchlist)
    st.session_state["watchlist"] = watchlist
    return True, f"Added '{title}' to your watchlist!"


def remove_from_watchlist(movie_id):
    """Remove a movie from the watchlist by movie_id."""
    try:
        target_id = int(movie_id)
    except Exception:
        return False

    watchlist = load_watchlist()
    updated = [x for x in watchlist if int(x.get("movieId", -1)) != target_id]
    save_watchlist(updated)
    st.session_state["watchlist"] = updated
    return True


def toggle_watched_status(movie_id):
    """Toggle watched status between True and False for a movie."""
    try:
        target_id = int(movie_id)
    except Exception:
        return False, "Invalid movie ID"

    watchlist = load_watchlist()
    found = False
    new_status = False
    movie_title = "Movie"

    for item in watchlist:
        if int(item.get("movieId", -1)) == target_id:
            current_status = bool(item.get("watched", False))
            new_status = not current_status
            item["watched"] = new_status
            movie_title = item.get("title", "Movie")
            found = True
            break

    if found:
        save_watchlist(watchlist)
        st.session_state["watchlist"] = watchlist
        status_text = "Watched" if new_status else "Not Watched"
        return True, f"Marked '{movie_title}' as {status_text}!"
    return False, "Movie not found in watchlist."


def extract_movie_year(movie):
    """Extract release year from movie title or TMDB details."""
    if "year" in movie and pd.notna(movie["year"]):
        try:
            return int(movie["year"])
        except (ValueError, TypeError):
            pass
    title = str(movie.get("title", ""))
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    tmdb_id = movie.get("tmdbId")
    if pd.notna(tmdb_id) and tmdb_id:
        try:
            details = get_movie_details(tmdb_id)
            if details and details.get("release_date"):
                year_str = str(details["release_date"])[:4]
                if year_str.isdigit():
                    return int(year_str)
        except Exception:
            pass
    return 0


def get_movie_rating(movie):
    """Get TMDB rating for sorting."""
    if "vote_average" in movie and movie["vote_average"] is not None:
        try:
            return float(movie["vote_average"])
        except (ValueError, TypeError):
            pass
    tmdb_id = movie.get("tmdbId")
    if pd.notna(tmdb_id) and tmdb_id:
        try:
            details = get_movie_details(tmdb_id)
            if details and details.get("vote_average") is not None:
                rating = float(details["vote_average"])
                movie["vote_average"] = rating
                return rating
        except Exception:
            pass
    return None


# ---------------- MOVIE CARD HTML ----------------

def _build_card_html(movie, score_column=None, rating_text=None, show_watched_badge=False):
    """Build an HTML movie card for one movie.

    Uses <span style='display:block'> instead of <div> because
    Streamlit's markdown sanitizer can break nested div structures,
    causing literal '</div>' text to leak into the rendered output.
    """

    poster = get_movie_poster(movie.get("tmdbId"))
    poster_url = poster if poster else PLACEHOLDER_POSTER

    title = movie.get("title", "Unknown")
    title_safe = (
        title
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

    genres = movie.get("genres", "")
    if genres == "(no genres listed)":
        genres = ""
    genres_display = genres.replace("|", " · ")

    score_html = ""
    if rating_text:
        score_html = (
            f'<span class="movie-score">'
            f'★ {rating_text}</span>'
        )
    elif score_column and score_column in movie:
        score_html = (
            f'<span class="movie-score">'
            f'★ {movie[score_column]:.3f}</span>'
        )

    watched_html = ""
    if show_watched_badge:
        if movie.get("watched", False):
            watched_html = '<span class="badge-watched">✓ Watched</span>'
        else:
            watched_html = '<span class="badge-unwatched">⏳ Not Watched</span>'

    return (
        '<span class="movie-card">'
            '<span class="poster-wrap">'
                f'<img src="{poster_url}" alt="{title_safe}" loading="lazy"'
                f" onerror=\"this.src='{PLACEHOLDER_POSTER}';\">"
            '</span>'
            '<span class="card-body">'
                f'<p class="movie-title" title="{title_safe}">{title_safe}</p>'
                f'<span class="movie-genre">{genres_display}</span>'
                f'{score_html}'
                f'{watched_html}'
            '</span>'
        '</span>'
    )


# ---------------- MOVIE GRID DISPLAY ----------------

def show_movies(data, score_column=None, key_prefix="grid"):
    """Display movies in a 5-column grid with View Details and Watchlist buttons."""

    if data.empty:
        st.info("No movies found.")
        return

    for start in range(0, len(data), 5):

        batch = data.iloc[start:start + 5]
        cols = st.columns(5)

        for col, (_, movie) in zip(cols, batch.iterrows()):

            with col:
                card_html = _build_card_html(
                    movie, score_column
                )
                st.markdown(
                    card_html,
                    unsafe_allow_html=True
                )

                movie_id = int(movie["movieId"])

                # View Details button
                if st.button(
                    "View Details",
                    key=f"{key_prefix}_detail_{movie_id}_{start}",
                    use_container_width=True
                ):
                    st.session_state[
                        "selected_movie_id"
                    ] = movie_id
                    st.rerun()

                # Add to Watchlist button
                if is_in_watchlist(movie_id):
                    st.button(
                        "✓ In Watchlist",
                        key=f"{key_prefix}_wl_in_{movie_id}_{start}",
                        disabled=True,
                        use_container_width=True
                    )
                else:
                    if st.button(
                        "➕ Add to Watchlist",
                        key=f"{key_prefix}_wl_add_{movie_id}_{start}",
                        use_container_width=True
                    ):
                        success, msg = add_to_watchlist(movie)
                        if success:
                            st.toast(msg, icon="🎬")
                            st.rerun()
                        else:
                            st.toast(msg, icon="ℹ️")


# ---------------- WATCHLIST GRID DISPLAY ----------------

def show_watchlist_movies(watchlist_items):
    """Display saved movies on the Watchlist page with Watched toggle and Remove buttons."""

    if not watchlist_items:
        st.info("No matching movies found in your watchlist.")
        return

    for start in range(0, len(watchlist_items), 5):

        batch = watchlist_items[start:start + 5]
        cols = st.columns(5)

        for col, movie in zip(cols, batch):

            with col:
                # Fetch TMDB rating if available
                rating_str = None
                tmdb_id = movie.get("tmdbId")
                if pd.notna(tmdb_id) and tmdb_id:
                    details = get_movie_details(tmdb_id)
                    if details and details.get("vote_average") is not None:
                        rating_str = f"{details['vote_average']:.1f}/10"
                        movie["vote_average"] = details["vote_average"]

                card_html = _build_card_html(
                    movie,
                    rating_text=rating_str,
                    show_watched_badge=True
                )
                st.markdown(
                    card_html,
                    unsafe_allow_html=True
                )

                movie_id = int(movie["movieId"])
                is_watched = bool(movie.get("watched", False))

                # View Details button
                if st.button(
                    "View Details",
                    key=f"wl_page_det_{movie_id}_{start}",
                    use_container_width=True
                ):
                    st.session_state["selected_movie_id"] = movie_id
                    st.rerun()

                # Action row: Toggle Watched button + Remove button
                act_col1, act_col2 = st.columns([1, 1])

                with act_col1:
                    watch_btn_label = "⏳ Unwatch" if is_watched else "✅ Watched"
                    if st.button(
                        watch_btn_label,
                        key=f"wl_toggle_{movie_id}_{start}",
                        use_container_width=True,
                        help="Click to toggle watched status"
                    ):
                        success, msg = toggle_watched_status(movie_id)
                        if success:
                            icon = "✅" if not is_watched else "⏳"
                            st.toast(msg, icon=icon)
                            st.rerun()

                with act_col2:
                    if st.button(
                        "🗑️ Remove",
                        key=f"wl_page_rem_{movie_id}_{start}",
                        use_container_width=True,
                        help="Remove from watchlist"
                    ):
                        remove_from_watchlist(movie_id)
                        st.toast(f"Removed '{movie.get('title', 'Movie')}' from your watchlist.", icon="🗑️")
                        st.rerun()


# ---------------- DETAIL PAGE RENDERER ----------------

def show_movie_detail(movie_row):
    """Display a full movie detail page."""

    tmdb_id = movie_row.get("tmdbId")
    local_title = movie_row.get("title", "Unknown")
    local_genres = movie_row.get("genres", "")
    m_id = int(movie_row["movieId"])

    # Action bar: Back button + Watchlist button + Watched toggle (if in watchlist)
    if is_in_watchlist(m_id):
        wl = st.session_state.get("watchlist", [])
        m_watched = any(int(w.get("movieId", -1)) == m_id and w.get("watched", False) for w in wl)

        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        with btn_col1:
            if st.button("← Back to Browse", use_container_width=True):
                st.session_state.pop("selected_movie_id", None)
                st.rerun()

        with btn_col2:
            st.button(
                "✓ In Watchlist",
                key=f"det_in_wl_{m_id}",
                disabled=True,
                use_container_width=True
            )

        with btn_col3:
            toggle_label = "⏳ Mark Not Watched" if m_watched else "✅ Mark as Watched"
            if st.button(toggle_label, key=f"det_toggle_wl_{m_id}", use_container_width=True):
                success, msg = toggle_watched_status(m_id)
                if success:
                    st.toast(msg, icon="✅" if not m_watched else "⏳")
                    st.rerun()
    else:
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("← Back to Browse", use_container_width=True):
                st.session_state.pop("selected_movie_id", None)
                st.rerun()

        with btn_col2:
            if st.button(
                "➕ Add to Watchlist",
                key=f"det_add_wl_{m_id}",
                use_container_width=True
            ):
                success, msg = add_to_watchlist(movie_row)
                if success:
                    st.toast(msg, icon="🎬")
                    st.rerun()
                else:
                    st.toast(msg, icon="ℹ️")

    st.markdown("---")

    # Fetch TMDB details with a spinner
    with st.spinner("Loading movie details..."):
        details = get_movie_details(tmdb_id)

    # Use TMDB data if available, fall back to local CSV data
    if details:
        title = details["title"]
        tagline = details["tagline"]
        overview = details["overview"]
        release_date = details["release_date"]
        runtime = details["runtime"]
        vote_avg = details["vote_average"]
        vote_count = details["vote_count"]
        genre_list = details["genres"]
        poster_url = details["poster_url"]
    else:
        title = local_title
        tagline = ""
        overview = "Details are currently unavailable."
        release_date = ""
        runtime = None
        vote_avg = None
        vote_count = 0
        genre_list = (
            [g.strip() for g in local_genres.split("|")]
            if local_genres
            and local_genres != "(no genres listed)"
            else []
        )
        poster_url = None

    # Escape title for HTML
    title_safe = (
        title
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

    # --- Layout: poster left, info right ---
    col_poster, col_info = st.columns([1, 2])

    with col_poster:
        img_url = poster_url if poster_url else PLACEHOLDER_POSTER
        st.markdown(
            f'<span class="poster-wrap">'
            f'<img src="{img_url}" alt="{title_safe}" '
            f'loading="lazy" '
            f"onerror=\"this.src='{PLACEHOLDER_POSTER}';\">"
            f'</span>',
            unsafe_allow_html=True
        )

    with col_info:
        # Title
        st.markdown(
            f'<p class="detail-title">{title_safe}</p>',
            unsafe_allow_html=True
        )

        # Tagline
        if tagline:
            tagline_safe = (
                tagline
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            st.markdown(
                f'<p class="detail-tagline">'
                f'"{tagline_safe}"</p>',
                unsafe_allow_html=True
            )

        # Meta badges (release date, runtime, rating)
        meta_parts = []

        if release_date:
            meta_parts.append(
                f'<span class="detail-meta-item">'
                f'📅 {release_date}</span>'
            )

        if runtime:
            hours = runtime // 60
            mins = runtime % 60
            rt_str = (
                f"{hours}h {mins}m" if hours
                else f"{mins}m"
            )
            meta_parts.append(
                f'<span class="detail-meta-item">'
                f'⏱ {rt_str}</span>'
            )

        if vote_avg is not None:
            meta_parts.append(
                f'<span class="detail-rating-badge">'
                f'★ {vote_avg:.1f}/10</span>'
            )
            if vote_count:
                meta_parts.append(
                    f'<span class="detail-meta-item">'
                    f'{vote_count:,} votes</span>'
                )

        if meta_parts:
            meta_html = (
                '<span class="detail-meta">'
                + "".join(meta_parts)
                + '</span>'
            )
            st.markdown(
                meta_html, unsafe_allow_html=True
            )

        # Genre tags
        if genre_list:
            genre_tags = "".join(
                f'<span class="detail-genre-tag">'
                f'{g}</span>'
                for g in genre_list
            )
            st.markdown(
                f'<span class="detail-genres">'
                f'{genre_tags}</span>',
                unsafe_allow_html=True
            )

        # Overview / Storyline
        st.markdown(
            '<p class="detail-overview-label">'
            'Storyline</p>',
            unsafe_allow_html=True
        )

        overview_safe = (
            overview
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        ) if overview else "No overview available."

        st.markdown(
            f'<p class="detail-overview">'
            f'{overview_safe}</p>',
            unsafe_allow_html=True
        )


# ---------------- WEBSITE HEADER ----------------

st.title("🎬 MovieFlix")
st.caption(f"Explore {len(movies):,} movies")

st.markdown(
    "### Discover movies you'll love."
)

st.divider()


# ---------------- SIDEBAR ----------------

st.sidebar.title("MovieFlix")
st.sidebar.caption("Personalized Movie Discovery")

# Ensure watchlist is loaded in session_state before sidebar count
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = load_watchlist()

saved_count = len(st.session_state.get("watchlist", []))

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Personalized Recommendations",
        "Find Similar Movies",
        "My Watchlist"
    ]
)

if saved_count > 0:
    st.sidebar.markdown(f"🔖 **Watchlist:** `{saved_count}` movie{'s' if saved_count != 1 else ''} saved")


# ---------------- GENRE LIST ----------------

@st.cache_data
def get_all_genres(_movies_df):
    """Extract all unique genres from the pipe-separated genres column."""
    all_genres = set()
    for genre_str in _movies_df["genres"].dropna():
        if genre_str and genre_str != "(no genres listed)":
            for g in genre_str.split("|"):
                g = g.strip()
                if g:
                    all_genres.add(g)
    return sorted(all_genres)

ALL_GENRES = get_all_genres(movies)


# ---------------- RELEASE YEARS LIST ----------------

@st.cache_data
def get_available_years(_movies_df):
    """Extract sorted unique valid release years in descending order."""
    if "year" in _movies_df:
        valid_years = _movies_df["year"].dropna().astype(int).unique()
        return sorted(valid_years, reverse=True)
    return []

AVAILABLE_YEARS = get_available_years(movies)


# ---------------- SESSION STATE INIT ----------------

if "selected_movie_id" not in st.session_state:
    st.session_state["selected_movie_id"] = None


# ---------------- MOVIE DETAIL VIEW ----------------
# If a movie is selected, show its detail page instead
# of the normal page content.

if st.session_state["selected_movie_id"] is not None:

    sel_id = st.session_state["selected_movie_id"]

    movie_match = movies[movies["movieId"] == sel_id]

    if movie_match.empty:
        st.warning("Movie not found in database.")
        if st.button("← Back to Browse"):
            st.session_state["selected_movie_id"] = None
            st.rerun()
    else:
        show_movie_detail(movie_match.iloc[0])


# ---------------- HOME PAGE ----------------

elif page == "Home":

    st.subheader("Welcome to MovieFlix")

    st.write(
        "Discover movies using machine learning, content-based search, "
        "and collaborative filtering."
    )

    # --- 1. Search & Filter Controls ---
    filter_col1, filter_col2, filter_col3 = st.columns([2.5, 1.3, 1.2])

    with filter_col1:
        search = st.text_input(
            "🔍 Search for a movie",
            placeholder="Search by title (e.g. Toy Story, Inception)...",
            key="home_search_input"
        )

    with filter_col2:
        selected_genre = st.selectbox(
            "🎭 Genre",
            ["All Genres"] + ALL_GENRES,
            key="home_genre_select"
        )

    with filter_col3:
        year_filter_type = st.selectbox(
            "📅 Release Year",
            ["All Years", "Specific Year", "Year Range"],
            key="home_year_type_select"
        )

    # Sub-controls for Specific Year or Year Range
    specific_year = None
    year_range = None

    if year_filter_type == "Specific Year":
        year_col, _ = st.columns([1.5, 3.5])
        with year_col:
            default_year_idx = AVAILABLE_YEARS.index(2020) if 2020 in AVAILABLE_YEARS else 0
            specific_year = st.selectbox(
                "Choose Year",
                options=AVAILABLE_YEARS,
                index=default_year_idx,
                key="home_specific_year_select"
            )

    elif year_filter_type == "Year Range":
        range_col, _ = st.columns([3, 2])
        with range_col:
            min_y = int(movies["year"].min()) if pd.notna(movies["year"].min()) else 1874
            max_y = int(movies["year"].max()) if pd.notna(movies["year"].max()) else 2023
            year_range = st.slider(
                "Choose Year Range",
                min_value=min_y,
                max_value=max_y,
                value=(1990, 2000),
                step=1,
                key="home_year_range_slider"
            )

    # --- 2. Track Filter State for Pagination Reset ---
    current_filter_state = (
        search.strip().lower(),
        selected_genre,
        year_filter_type,
        specific_year,
        year_range
    )

    if "home_active_filters" not in st.session_state:
        st.session_state["home_active_filters"] = current_filter_state
        st.session_state["home_page_num"] = 1

    # Reset to page 1 ONLY when search, genre, or year filters change
    if st.session_state["home_active_filters"] != current_filter_state:
        st.session_state["home_page_num"] = 1
        st.session_state["home_active_filters"] = current_filter_state

    # --- 3. Apply Filters ---
    filtered = movies.copy()

    # Search filter
    if search.strip():
        filtered = filtered[
            filtered["title"].str.contains(
                search.strip(),
                case=False,
                na=False
            )
        ]

    # Genre filter
    if selected_genre != "All Genres":
        filtered = filtered[
            filtered["genres"].str.contains(
                selected_genre,
                case=False,
                na=False
            )
        ]

    # Year filter
    if year_filter_type == "Specific Year" and specific_year is not None:
        filtered = filtered[filtered["year"] == specific_year]
    elif year_filter_type == "Year Range" and year_range is not None:
        start_y, end_y = year_range
        filtered = filtered[
            filtered["year"].notna() &
            (filtered["year"] >= start_y) &
            (filtered["year"] <= end_y)
        ]

    # Sort filtered results: prioritize popular titles (from bayesian_pop_df) then alphabetical
    pop_score_map = bayesian_pop_df.set_index("movieId")["wr_score"]
    filtered["_pop_score"] = filtered["movieId"].map(pop_score_map).fillna(0.0)
    filtered = filtered.sort_values(by=["_pop_score", "movieId"], ascending=[False, True])

    # --- 4. Pagination Setup ---
    PAGE_SIZE = 20
    total_movies = len(filtered)
    total_pages = max(1, (total_movies + PAGE_SIZE - 1) // PAGE_SIZE)

    # Ensure page number stays within valid [1, total_pages] boundaries
    current_page = st.session_state.get("home_page_num", 1)
    if current_page > total_pages:
        current_page = total_pages
        st.session_state["home_page_num"] = current_page
    elif current_page < 1:
        current_page = 1
        st.session_state["home_page_num"] = current_page

    # --- 5. Display Active Filter Summary & Result Count ---
    active_filters_desc = []
    if search.strip():
        active_filters_desc.append(f'Title containing "{search.strip()}"')
    if selected_genre != "All Genres":
        active_filters_desc.append(f"Genre: **{selected_genre}**")
    if year_filter_type == "Specific Year" and specific_year is not None:
        active_filters_desc.append(f"Year: **{specific_year}**")
    elif year_filter_type == "Year Range" and year_range is not None:
        active_filters_desc.append(f"Years: **{year_range[0]}–{year_range[1]}**")

    if active_filters_desc:
        summary_str = " · ".join(active_filters_desc)
        st.caption(f"Filtered by: {summary_str} — Found **{total_movies:,}** movie{'s' if total_movies != 1 else ''}")
    else:
        st.caption(f"Browsing **{total_movies:,}** movies in catalog (sorted by popularity)")

    # --- 6. Slice and Render Current Page Movies ---
    if total_movies == 0:
        st.info("No movies found matching your selected filters. Try broadening your search or adjusting the filters.")
    else:
        start_idx = (current_page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_movies)
        page_movies = filtered.iloc[start_idx:end_idx]

        show_movies(page_movies, key_prefix="home")

        # --- 7. Pagination Controls ---
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        col_prev, col_info, col_next = st.columns([1, 2, 1])

        with col_prev:
            prev_disabled = (current_page <= 1)
            if st.button("⬅️ Previous", disabled=prev_disabled, use_container_width=True, key="home_prev_btn"):
                st.session_state["home_page_num"] = max(1, current_page - 1)
                st.rerun()

        with col_info:
            st.markdown(
                f"<div style='text-align: center; line-height: 38px; font-size: 15px; font-weight: 600; color: #e5e5e5;'>"
                f"Page <span style='color: #e50914;'>{current_page:,}</span> of <span style='color: #ffffff;'>{total_pages:,}</span>"
                f"<span style='display: block; font-size: 12px; color: #888888; font-weight: 400;'>"
                f"Showing {start_idx + 1:,}–{end_idx:,} of {total_movies:,} movies</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col_next:
            next_disabled = (current_page >= total_pages)
            if st.button("Next ➡️", disabled=next_disabled, use_container_width=True, key="home_next_btn"):
                st.session_state["home_page_num"] = min(total_pages, current_page + 1)
                st.rerun()


# ---------------- PERSONALIZED PAGE ----------------

elif page == "Personalized Recommendations":

    st.subheader("Your Personal Recommendations")

    user_mode = st.radio(
        "Recommendation Profile",
        ["Existing MovieLens User", "✨ New User (Cold-Start)"],
        horizontal=True,
        help="Choose an existing user profile with rating history, or start as a new user with customizable taste preferences."
    )

    if user_mode == "Existing MovieLens User":
        user_ids = train_matrix.index.tolist()

        selected_user = st.selectbox(
            "Select a MovieLens User ID",
            user_ids
        )

        if st.button("Get Recommendations", key="btn_existing_recs"):
            with st.spinner("Generating hybrid recommendations..."):
                recommendations = hybrid_recommend(
                    selected_user, 10
                )

            st.subheader("Recommended For You")
            show_movies(
                recommendations,
                "hybrid_score",
                key_prefix="recs"
            )

    else:
        # --- Cold-Start New User Profile ---
        st.markdown(
            "Welcome to MovieFlix! Tell us what you like to receive personalized recommendations "
            "even without an existing MovieLens profile."
        )

        # Load persisted preferences
        stored_prefs = load_user_preferences()

        # Genre preferences
        saved_genres = [g for g in stored_prefs.get("favourite_genres", []) if g in ALL_GENRES]
        fav_genres = st.multiselect(
            "🎭 Select your favourite genres:",
            options=ALL_GENRES,
            default=saved_genres,
            help="Select one or more genres to tailor recommendations to your taste."
        )

        # Favourite movies from catalog
        popular_catalog_titles = movies[movies["movieId"].isin(bayesian_pop_df["movieId"].head(500))]["title"].dropna().tolist()
        all_cf_titles = movies[movies["movieId"].isin(train_matrix.columns)]["title"].dropna().tolist()
        title_options = list(dict.fromkeys(popular_catalog_titles + all_cf_titles))

        title_to_id = dict(zip(movies["title"], movies["movieId"]))
        id_to_title = dict(zip(movies["movieId"], movies["title"]))

        saved_fav_titles = [id_to_title[mid] for mid in stored_prefs.get("favourite_movie_ids", []) if mid in id_to_title]
        fav_movies = st.multiselect(
            "🎬 Select a few favourite movies you love:",
            options=title_options,
            default=saved_fav_titles,
            help="Select movies from our catalog. We'll use TF-IDF content similarity to find movies like them."
        )

        # Optional ratings to demonstrate smooth transition to Collaborative Filtering
        st.markdown("##### ⭐ Quick Ratings (Optional - Smooth Transition to Collaborative Filtering)")
        st.caption(
            "Rate 1 to 5 movies (0.5 to 5.0 stars). Providing 3–5 ratings smoothly transitions your recommendations "
            "from cold-start preferences to collaborative filtering!"
        )

        saved_ratings = stored_prefs.get("user_ratings", {})

        # Let user rate selected movies, or sample iconic movies if none selected
        user_ratings_input = {}
        rated_candidate_titles = list(fav_movies)
        if not rated_candidate_titles:
            sample_ids = [1, 296, 260]  # Toy Story, Pulp Fiction, Star Wars
            rated_candidate_titles = [id_to_title[m] for m in sample_ids if m in id_to_title]

        rate_cols = st.columns(min(len(rated_candidate_titles), 4)) if rated_candidate_titles else []
        for i, (col, m_title) in enumerate(zip(rate_cols, rated_candidate_titles[:4])):
            with col:
                m_id = title_to_id.get(m_title)
                default_val = float(saved_ratings.get(m_id, 4.5)) if m_id else 4.5
                rating_val = st.slider(
                    f"{m_title[:20]}...",
                    min_value=0.5,
                    max_value=5.0,
                    value=default_val,
                    step=0.5,
                    key=f"cold_rate_{m_id}_{i}"
                )
                if m_id:
                    user_ratings_input[m_id] = rating_val

        # Action buttons
        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            get_cold_recs = st.button("✨ Get Personalized Recommendations", key="btn_cold_recs", use_container_width=True)
        with btn_col2:
            save_pref_btn = st.button("💾 Save Preferences", key="btn_save_prefs", use_container_width=True)

        selected_fav_ids = [title_to_id[t] for t in fav_movies if t in title_to_id]

        if save_pref_btn:
            prefs_to_save = {
                "favourite_genres": fav_genres,
                "favourite_movie_ids": selected_fav_ids,
                "user_ratings": user_ratings_input
            }
            if save_user_preferences(prefs_to_save):
                st.toast("Your preferences have been saved!", icon="💾")
            else:
                st.error("Failed to save preferences.")

        # If user clicks button, recompute; otherwise initialize with initial recommendations (never empty!)
        if get_cold_recs or "cold_start_results" not in st.session_state:
            with st.spinner("Generating personalized recommendations..."):
                recs = recommend_cold_start_user(
                    favourite_genres=fav_genres,
                    favourite_movie_ids=selected_fav_ids,
                    user_ratings=user_ratings_input if get_cold_recs else saved_ratings,
                    n=10
                )
                st.session_state["cold_start_results"] = recs
                st.session_state["cold_start_alpha"] = recs["alpha"].iloc[0] if "alpha" in recs.columns and not recs.empty else 0.0
                st.session_state["cold_start_num_ratings"] = len(user_ratings_input) if get_cold_recs else len(saved_ratings)
        else:
            recs = st.session_state["cold_start_results"]

        # Display transition mode status
        k_ratings = st.session_state.get("cold_start_num_ratings", len(user_ratings_input))
        alpha = st.session_state.get("cold_start_alpha", 0.0)

        if k_ratings == 0 or alpha == 0.0:
            if fav_genres and selected_fav_ids:
                st.info("✨ **Cold-Start Mode Active**: 45% Content-Based + 35% Genre Alignment + 20% Bayesian Popularity")
            elif fav_genres:
                st.info("✨ **Cold-Start Mode Active**: 70% Genre Preference + 30% Bayesian Popularity")
            elif selected_fav_ids:
                st.info("✨ **Cold-Start Mode Active**: 75% Content Similarity + 25% Bayesian Popularity")
            else:
                st.info("✨ **Initial Popularity Mode Active**: Displaying top-rated popular movies using IMDb Bayesian Weighted Rating.")
        else:
            cf_pct = int(alpha * 100)
            cs_pct = 100 - cf_pct
            st.success(
                f"⚡ **Adaptive Hybrid Transition Active**: {cf_pct}% Collaborative Filtering + "
                f"{cs_pct}% Cold-Start Preferences (Based on {k_ratings} user rating{'s' if k_ratings != 1 else ''})"
            )

        st.subheader("Recommended For You")
        show_movies(recs, "score", key_prefix="cold_recs")


# ---------------- SIMILAR MOVIES PAGE ----------------

elif page == "Find Similar Movies":

    st.subheader("Find Movies Similar To Your Favourite")
    st.write(
        "Search our 87,000+ catalog by full or partial title (e.g. *Krrish*, *Toy Story*, *The Dark Knight*) "
        "to discover movies with similar storylines, themes, and genres."
    )

    # Search bar input
    col_search_input, col_search_btn = st.columns([4, 1])
    with col_search_input:
        search_query = st.text_input(
            "🔍 Search movie title",
            placeholder="Type a movie title (e.g. Krrish, Toy Story, The Dark Knight)...",
            key="similar_search_input"
        )
    with col_search_btn:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        do_search = st.button("Search", use_container_width=True, key="similar_search_btn")

    # If query is provided, execute fast cached lookup
    if search_query.strip():
        with st.spinner("Searching movie catalog..."):
            matching_movies = search_movies_by_title(search_query.strip(), limit=50)

        if matching_movies.empty:
            st.warning(f"No movies found matching '{search_query.strip()}'. Please check your spelling or try another keyword.")
        else:
            match_options = {}
            for _, row in matching_movies.iterrows():
                genre_str = f" — {row['genres'].replace('|', ', ')}" if pd.notna(row["genres"]) and row["genres"] != "(no genres listed)" else ""
                label = f"{row['title']}{genre_str}"
                match_options[label] = int(row["movieId"])

            st.caption(f"Found **{len(matching_movies)}** matching title{'s' if len(matching_movies) != 1 else ''}:")
            selected_label = st.selectbox(
                "Select the movie you love:",
                options=list(match_options.keys()),
                key="similar_movie_select"
            )
            selected_movie_id = match_options[selected_label]
            selected_title_clean = selected_label.split(" — ")[0]

            if st.button("✨ Find Similar Movies", use_container_width=True, key="btn_run_similar"):
                with st.spinner(f"Finding movies similar to '{selected_title_clean}'..."):
                    recommendations = recommend_similar_movies(
                        selected_movie_id, n=10
                    )
                    st.session_state["similar_results"] = recommendations
                    st.session_state["similar_source_id"] = selected_movie_id
                    st.session_state["similar_source_title"] = selected_title_clean

    else:
        st.info("💡 Enter a movie title above and press Enter (or click Search) to get started.")

    # Render results from session_state so they persist across Watchlist / Detail interactions
    if "similar_results" in st.session_state and not st.session_state["similar_results"].empty:
        source_title = st.session_state.get("similar_source_title", "Selected Movie")
        st.divider()
        st.subheader(f"🎬 Movies Similar to '{source_title}'")
        show_movies(
            st.session_state["similar_results"],
            "similarity_score",
            key_prefix="similar"
        )


# ---------------- MY WATCHLIST PAGE ----------------

elif page == "My Watchlist":

    st.subheader("🔖 My Watchlist")

    current_watchlist = load_watchlist()
    st.session_state["watchlist"] = current_watchlist

    if not current_watchlist:
        st.info(
            "Your watchlist is currently empty.\n\n"
            "Explore movies on the **Home**, **Personalized Recommendations**, or **Find Similar Movies** "
            "pages and click **➕ Add to Watchlist** to save them here!"
        )
    else:
        # --- 1. Watchlist Statistics ---
        total_count = len(current_watchlist)
        watched_count = sum(1 for m in current_watchlist if m.get("watched", False))
        remaining_count = total_count - watched_count

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric(label="🎬 Total Movies", value=total_count)
        with stat_col2:
            st.metric(label="✅ Watched", value=watched_count)
        with stat_col3:
            st.metric(label="⏳ Remaining to Watch", value=remaining_count)

        st.divider()

        # --- 2. Search, Sorting & Status Filters ---
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 2, 2])

        with ctrl_col1:
            search_query = st.text_input(
                "🔍 Search saved movies",
                placeholder="Search by title...",
                key="watchlist_search_input"
            )

        with ctrl_col2:
            sort_option = st.selectbox(
                "Sort by",
                [
                    "Highest rated",
                    "Lowest rated",
                    "Movie title (A-Z)",
                    "Release year (Newest first)"
                ],
                key="watchlist_sort_select"
            )

        with ctrl_col3:
            status_filter = st.selectbox(
                "Filter by status",
                [
                    "All Movies",
                    "Not Watched",
                    "Watched"
                ],
                key="watchlist_status_select"
            )

        # Apply Status filter
        display_items = list(current_watchlist)
        if status_filter == "Watched":
            display_items = [m for m in display_items if m.get("watched", False)]
        elif status_filter == "Not Watched":
            display_items = [m for m in display_items if not m.get("watched", False)]

        # Apply Search filter
        if search_query.strip():
            q = search_query.strip().lower()
            display_items = [
                m for m in display_items
                if q in m.get("title", "").lower()
            ]

        # Apply Sorting
        if sort_option == "Highest rated":
            display_items.sort(
                key=lambda m: (
                    1 if get_movie_rating(m) is not None else 0,
                    get_movie_rating(m) if get_movie_rating(m) is not None else -1.0
                ),
                reverse=True
            )
        elif sort_option == "Lowest rated":
            display_items.sort(
                key=lambda m: (
                    0 if get_movie_rating(m) is not None else 1,
                    get_movie_rating(m) if get_movie_rating(m) is not None else 999.0
                )
            )
        elif sort_option == "Movie title (A-Z)":
            display_items.sort(
                key=lambda m: m.get("title", "").strip().lower()
            )
        elif sort_option == "Release year (Newest first)":
            display_items.sort(
                key=extract_movie_year,
                reverse=True
            )

        st.caption(f"Showing **{len(display_items)}** of **{total_count}** saved movie(s)")

        col_clear, _ = st.columns([1, 4])
        with col_clear:
            if st.button("🗑️ Clear Entire Watchlist"):
                save_watchlist([])
                st.session_state["watchlist"] = []
                st.toast("Watchlist cleared.", icon="🗑️")
                st.rerun()

        # Display Grid
        show_watchlist_movies(display_items)


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "MovieFlix | Built with Python, Pandas, "
    "Scikit-learn, Streamlit and TMDB."
)
