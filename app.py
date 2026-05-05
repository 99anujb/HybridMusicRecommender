import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tunewise", page_icon="🎵", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Background */
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important; padding: 8px 0 !important;
}

/* All text white-ish */
h1, h2, h3, h4, p, label, .stMarkdown, div[class*="stText"] { color: #f0f0f0 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 16px;
    backdrop-filter: blur(10px);
}
div[data-testid="metric-container"] label { color: #a0a0c0 !important; font-size: 13px !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 28px !important; font-weight: 700 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 12px 40px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(102,126,234,0.6) !important;
}

/* Selectbox & slider */
.stSelectbox > div > div, .stSlider {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}

/* Dataframe */
.stDataFrame { border-radius: 16px !important; overflow: hidden; }

/* Success / info boxes */
.stSuccess, .stInfo { border-radius: 12px !important; }

/* Song card */
.song-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 12px;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease;
}
.song-card:hover { background: rgba(255,255,255,0.11); transform: translateY(-2px); }
.song-rank { font-size: 28px; font-weight: 800; color: #667eea; margin-right: 12px; }
.song-title { font-size: 17px; font-weight: 600; color: #ffffff; }
.song-artist { font-size: 13px; color: #a0a0c0; margin-top: 2px; }
.song-meta { font-size: 12px; color: #7070a0; margin-top: 6px; }
.song-pop-bar { height: 4px; border-radius: 2px; background: linear-gradient(90deg, #667eea, #f093fb); margin-top: 10px; }

/* Hero */
.hero {
    text-align: center;
    padding: 60px 20px 40px;
    background: rgba(255,255,255,0.03);
    border-radius: 24px;
    margin-bottom: 40px;
}
.hero h1 { font-size: 56px !important; font-weight: 800 !important;
            background: linear-gradient(135deg, #667eea, #f093fb);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { font-size: 18px !important; color: #a0a0c0 !important; max-width: 600px; margin: 0 auto; }

/* Vibe chips */
.chip {
    display: inline-block;
    background: rgba(102,126,234,0.2);
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 12px;
    color: #a0c0ff;
    margin: 2px;
}

/* Section headers */
.section-title {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 30px 0 16px !important;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(102,126,234,0.4);
}

/* Input labels */
.stSelectbox label, .stSlider label { color: #c0c0e0 !important; font-weight: 500 !important; }

/* Plots background */
.stPlot, [data-testid="stImage"] { border-radius: 16px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
AUDIO_FEATURES = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                   'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
BASELINE_FEATURES = AUDIO_FEATURES + ['release_year']
ENHANCED_FEATURES = BASELINE_FEATURES + [
    'key', 'mode', 'explicit', 'duration_min', 'decade',
    'energy_dance', 'valence_energy', 'acousticness_energy', 'log_duration'
]

MOOD_EMOJI = {"happy": "😊", "chill": "😌", "sad": "😢", "hype": "🔥"}
ERA_LABEL  = {"any": "Any Era", "90s": "90s", "2000s": "2000s", "2010s": "2010s"}

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_spotify_data.csv')
    df['duration_min'] = df['duration_ms'] / 60000
    df['decade'] = (df['release_year'] // 10) * 10
    df['energy_dance'] = df['energy'] * df['danceability']
    df['valence_energy'] = df['valence'] * df['energy']
    df['acousticness_energy'] = df['acousticness'] * (1 - df['energy'])
    df['log_duration'] = np.log1p(df['duration_ms'])
    df_sampled = df.sample(n=10000, random_state=42).reset_index(drop=True)
    return df, df_sampled

@st.cache_resource
def train_models(df):
    X = df[ENHANCED_FEATURES]
    y = df['popularity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb  = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
    lgbm = lgb.LGBMRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                               num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                               min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                               random_state=42, verbose=-1)
    rf   = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)

    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    stack = StackingRegressor(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
        final_estimator=Ridge(alpha=1.0), cv=3
    )
    stack.fit(X_train, y_train)

    def mets(model):
        p = model.predict(X_test)
        return dict(RMSE=np.sqrt(mean_squared_error(y_test, p)),
                    MAE=mean_absolute_error(y_test, p),
                    R2=r2_score(y_test, p),
                    Acc10=(np.abs(p - y_test) <= 10).mean() * 100,
                    preds=p)

    metrics = {
        'XGBoost': mets(xgb),
        'LightGBM': mets(lgbm),
        'Random Forest': mets(rf),
        'Stacking Ensemble': mets(stack),
    }
    return xgb, lgbm, rf, stack, X_test, y_test, metrics

@st.cache_resource
def build_recommender(df_sampled, _stack):
    scaler = MinMaxScaler()
    fm = scaler.fit_transform(df_sampled[AUDIO_FEATURES])
    nn = NearestNeighbors(n_neighbors=11, metric='cosine')
    nn.fit(fm)
    ds = df_sampled.copy()
    ds['predicted_popularity'] = _stack.predict(ds[ENHANCED_FEATURES])
    return nn, fm, ds

def hybrid_recommend(song_idx, nn, fm, ds, n=5, alpha=0.7):
    dists, idxs = nn.kneighbors([fm[song_idx]], n_neighbors=n + 1)
    dists, idxs = dists[0][1:], idxs[0][1:]
    scores = sorted(
        [(i, alpha * ds.iloc[i]['predicted_popularity'] / 100 + (1 - alpha) * (1 - d))
         for i, d in zip(idxs, dists)],
        key=lambda x: x[1], reverse=True
    )
    return [ds.iloc[i] for i, _ in scores[:n]]

def song_cards(recs):
    for rank, row in enumerate(recs, 1):
        pop = int(row['popularity'])
        pred = row['predicted_popularity']
        artist = str(row['artists']).strip("[]'\"")
        year = int(row['release_year'])
        bar_w = int(pop)
        st.markdown(f"""
        <div class="song-card">
            <div style="display:flex;align-items:center;">
                <span class="song-rank">#{rank}</span>
                <div style="flex:1">
                    <div class="song-title">{row['name']}</div>
                    <div class="song-artist">{artist}</div>
                    <div class="song-meta">📅 {year} &nbsp;·&nbsp; 🔥 Popularity: {pop}/100 &nbsp;·&nbsp; 🤖 Predicted: {pred:.0f}/100</div>
                    <div class="song-pop-bar" style="width:{bar_w}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Starting up Tunewise..."):
    df, df_sampled = load_data()
    xgb, lgbm, rf, stack, X_test, y_test, metrics = train_models(df)
    nn, fm, ds = build_recommender(df_sampled, stack)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Tunewise")
    st.markdown("<p style='color:#7070a0;font-size:13px'>Hybrid Music Recommender</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "🏠  Home",
        "🎧  Find My Vibe",
        "🔍  Songs Like This",
        "📊  How It Works",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<p style='color:#505070;font-size:12px'>Group 13 · ML Project<br>93,185 Spotify tracks<br>XGBoost + LightGBM + RF</p>", unsafe_allow_html=True)

# ── Home ──────────────────────────────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown("""
    <div class="hero">
        <h1>Tunewise 🎵</h1>
        <p>Discover music you'll love — powered by AI that understands your vibe, not just your history.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracks Analyzed", "93,185")
    c2.metric("Prediction Accuracy", f"{metrics['Stacking Ensemble']['Acc10']:.1f}%")
    c3.metric("AI Models Blended", "3")
    c4.metric("Audio Features", "9")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="song-card">
            <div style="font-size:32px">🎧</div>
            <div class="song-title" style="margin-top:10px">Find My Vibe</div>
            <div class="song-artist" style="margin-top:6px">Tell us your mood, tempo, and era — we'll pick the perfect seed song and recommend similar tracks.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="song-card">
            <div style="font-size:32px">🔍</div>
            <div class="song-title" style="margin-top:10px">Songs Like This</div>
            <div class="song-artist" style="margin-top:6px">Already have a song in mind? Search for it and get 5–10 sonically similar recommendations instantly.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">How recommendations work</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
        <div class="song-card" style="text-align:center">
            <div style="font-size:36px">🧠</div>
            <div class="song-title" style="margin-top:8px">AI Popularity Score</div>
            <div class="song-artist" style="margin-top:6px">Three ML models predict how popular a song is based on its audio DNA.</div>
        </div>
        <div class="song-card" style="text-align:center">
            <div style="font-size:36px">🎼</div>
            <div class="song-title" style="margin-top:8px">Audio Similarity</div>
            <div class="song-artist" style="margin-top:6px">We compare energy, danceability, tempo & 6 other features to find sonic twins.</div>
        </div>
        <div class="song-card" style="text-align:center">
            <div style="font-size:36px">⚖️</div>
            <div class="song-title" style="margin-top:8px">Hybrid Blend</div>
            <div class="song-artist" style="margin-top:6px">Both signals blend into one score — you control the balance with the α slider.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Find My Vibe ──────────────────────────────────────────────────────────────
elif page == "🎧  Find My Vibe":
    st.markdown('<h1 style="color:#fff;font-weight:800">Find My Vibe 🎧</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a0a0c0">Tell us how you feel and we\'ll find music that matches.</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-title">Your Mood</p>', unsafe_allow_html=True)
    mood_col = st.columns(4)
    mood_options = {"😊 Happy": "happy", "😌 Chill": "chill", "😢 Sad": "sad", "🔥 Hype": "hype"}
    if 'mood' not in st.session_state:
        st.session_state.mood = "happy"
    for col, (label, val) in zip(mood_col, mood_options.items()):
        if col.button(label, key=f"mood_{val}",
                      type="primary" if st.session_state.mood == val else "secondary"):
            st.session_state.mood = val

    st.markdown('<p class="section-title">Preferences</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    tempo     = c1.selectbox("Tempo", ["Any", "Slow 🐢", "Medium 🚶", "Fast ⚡"])
    era       = c2.selectbox("Era", ["Any Era 🌐", "90s 📼", "2000s 💿", "2010s 📱"])
    danceable = c3.selectbox("Danceability", ["Don't care", "Yes, make me move 💃", "No, something mellow"])

    st.markdown('<p class="section-title">Fine-tune</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    n_recs = c1.slider("How many recommendations?", 3, 10, 5)
    alpha  = c2.slider("Popularity vs. Similarity", 0.0, 1.0, 0.7, 0.05,
                       help="Left = pure audio similarity · Right = pure popularity")
    c2.caption(f"{'More popularity-focused' if alpha > 0.6 else 'More similarity-focused' if alpha < 0.4 else 'Balanced blend'}")

    st.markdown("")
    if st.button("Find My Music ✨"):
        df_f = ds.copy()

        mood_val = st.session_state.mood
        if mood_val == "happy":   df_f = df_f[(df_f['valence'] > 0.6) & (df_f['energy'] > 0.5)]
        elif mood_val == "chill": df_f = df_f[(df_f['valence'] > 0.4) & (df_f['energy'] < 0.5)]
        elif mood_val == "sad":   df_f = df_f[df_f['valence'] < 0.4]
        elif mood_val == "hype":  df_f = df_f[df_f['energy'] > 0.7]

        if "Slow" in tempo:   df_f = df_f[df_f['tempo'] < 0.4]
        elif "Medium" in tempo: df_f = df_f[(df_f['tempo'] >= 0.4) & (df_f['tempo'] <= 0.65)]
        elif "Fast" in tempo:   df_f = df_f[df_f['tempo'] > 0.65]

        if "Yes" in danceable:  df_f = df_f[df_f['danceability'] > 0.5]
        elif "mellow" in danceable: df_f = df_f[df_f['danceability'] <= 0.5]

        if "90s" in era:   df_f = df_f[(df_f['release_year'] >= 1990) & (df_f['release_year'] < 2000)]
        elif "2000s" in era: df_f = df_f[(df_f['release_year'] >= 2000) & (df_f['release_year'] < 2010)]
        elif "2010s" in era: df_f = df_f[(df_f['release_year'] >= 2010) & (df_f['release_year'] < 2020)]

        df_f = df_f.reset_index(drop=True)

        if df_f.empty:
            st.warning("No songs matched those filters. Try loosening your preferences.")
        else:
            seed = df_f.sample(1, random_state=42).iloc[0]
            match = ds[ds['name'] == seed['name']]
            song_idx = match.index[0] if not match.empty else ds.sample(1, random_state=42).index[0]

            seed_artist = str(seed['artists']).strip("[]'\"")
            st.markdown(f"""
            <div class="song-card" style="border-color:rgba(102,126,234,0.5);background:rgba(102,126,234,0.1)">
                <div style="font-size:13px;color:#667eea;font-weight:600;margin-bottom:6px">SEED SONG · {MOOD_EMOJI.get(mood_val,'')} {mood_val.upper()}</div>
                <div class="song-title">🎵 {seed['name']}</div>
                <div class="song-artist">{seed_artist} · {int(seed['release_year'])}</div>
            </div>
            """, unsafe_allow_html=True)

            recs = hybrid_recommend(song_idx, nn, fm, ds, n=n_recs, alpha=alpha)
            st.markdown(f'<p class="section-title">Your {n_recs} Recommendations</p>', unsafe_allow_html=True)
            song_cards(recs)

# ── Songs Like This ───────────────────────────────────────────────────────────
elif page == "🔍  Songs Like This":
    st.markdown('<h1 style="color:#fff;font-weight:800">Songs Like This 🔍</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a0a0c0">Pick a song you love — we\'ll find its sonic twins.</p>', unsafe_allow_html=True)

    search = st.selectbox("Search for a song", ds['name'].tolist(), label_visibility="collapsed",
                          placeholder="Type a song name...")

    c1, c2 = st.columns(2)
    n_recs = c1.slider("How many recommendations?", 3, 10, 5)
    alpha  = c2.slider("Popularity vs. Similarity", 0.0, 1.0, 0.7, 0.05)
    c2.caption(f"{'More popularity-focused' if alpha > 0.6 else 'More similarity-focused' if alpha < 0.4 else 'Balanced blend'}")

    if st.button("Find Similar Songs 🎵"):
        match = ds[ds['name'] == search]
        if match.empty:
            st.error("Song not found in catalog.")
        else:
            idx = match.index[0]
            row = ds.iloc[idx]
            artist = str(row['artists']).strip("[]'\"")

            # Seed card
            st.markdown(f"""
            <div class="song-card" style="border-color:rgba(102,126,234,0.5);background:rgba(102,126,234,0.1)">
                <div style="font-size:13px;color:#667eea;font-weight:600;margin-bottom:6px">YOUR PICK</div>
                <div class="song-title">🎵 {row['name']}</div>
                <div class="song-artist">{artist} · {int(row['release_year'])}</div>
                <div class="song-meta" style="margin-top:10px">
                    🔥 Popularity: {int(row['popularity'])}/100 &nbsp;·&nbsp;
                    ⚡ Energy: {row['energy']:.0%} &nbsp;·&nbsp;
                    💃 Danceability: {row['danceability']:.0%} &nbsp;·&nbsp;
                    😊 Mood: {row['valence']:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Audio profile chart
            feats  = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                      'liveness', 'speechiness', 'valence']
            labels = ['Acoustic', 'Danceable', 'Energy', 'Instrumental', 'Live', 'Speechy', 'Mood']
            vals   = [row[f] for f in feats]

            fig, ax = plt.subplots(figsize=(8, 2.5))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            colors = ['#667eea' if v > 0.5 else '#404060' for v in vals]
            bars = ax.barh(labels, vals, color=colors, height=0.6, edgecolor='none')
            ax.set_xlim(0, 1)
            ax.tick_params(colors='#a0a0c0', labelsize=11)
            ax.spines[:].set_visible(False)
            ax.xaxis.set_visible(False)
            for bar, v in zip(bars, vals):
                ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{v:.0%}', va='center', color='#a0a0c0', fontsize=10)
            st.pyplot(fig); plt.close()

            recs = hybrid_recommend(idx, nn, fm, ds, n=n_recs, alpha=alpha)
            st.markdown(f'<p class="section-title">Songs Like "{row["name"][:30]}"</p>', unsafe_allow_html=True)
            song_cards(recs)

# ── How It Works ─────────────────────────────────────────────────────────────
elif page == "📊  How It Works":
    st.markdown('<h1 style="color:#fff;font-weight:800">Under the Hood 📊</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a0a0c0">For the curious — how the AI makes its recommendations.</p>', unsafe_allow_html=True)

    # Model comparison
    st.markdown('<p class="section-title">Model Comparison</p>', unsafe_allow_html=True)
    rows = []
    for name, m in metrics.items():
        rows.append({'Model': name,
                     'Error (RMSE)': f"{m['RMSE']:.2f}",
                     'Avg Error (MAE)': f"{m['MAE']:.2f}",
                     'Variance Explained (R²)': f"{m['R2']:.4f}",
                     'Within 10 pts': f"{m['Acc10']:.1f}%"})
    st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)
    st.caption("Stacking Ensemble (XGBoost + LightGBM + Random Forest) powers all recommendations.")

    # Feature importance
    st.markdown('<p class="section-title">What Matters Most?</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    friendly_names = {
        'acousticness': 'Acoustic Level', 'danceability': 'Danceability',
        'energy': 'Energy', 'instrumentalness': 'Instrumental Level',
        'liveness': 'Live Feel', 'loudness': 'Loudness',
        'speechiness': 'Speechiness', 'tempo': 'Tempo',
        'valence': 'Positivity', 'release_year': 'Release Year',
        'key': 'Musical Key', 'mode': 'Major/Minor',
        'explicit': 'Explicit Content', 'duration_min': 'Song Duration',
        'decade': 'Decade', 'energy_dance': 'Energy × Dance',
        'valence_energy': 'Positivity × Energy',
        'acousticness_energy': 'Acoustic × Low Energy',
        'log_duration': 'Log Duration'
    }

    for col, (model, model_obj, color) in zip([c1, c2], [
        ('XGBoost', xgb, '#667eea'), ('LightGBM', lgbm, '#f093fb')
    ]):
        fi = pd.Series(model_obj.feature_importances_, index=ENHANCED_FEATURES)
        fi.index = [friendly_names.get(f, f) for f in fi.index]
        fi = fi.sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        ax.barh(fi.index, fi.values, color=color, edgecolor='none', height=0.6)
        ax.tick_params(colors='#a0a0c0', labelsize=10)
        ax.spines[:].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_title(model, color='#ffffff', fontsize=13, fontweight='bold')
        col.pyplot(fig); plt.close()

    # ROC curves
    st.markdown('<p class="section-title">Classification Performance</p>', unsafe_allow_html=True)
    median_pop = df['popularity'].median()
    y_cls = (y_test >= median_pop).astype(int)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    colors = ['#667eea', '#f093fb', '#43e97b', '#fa709a']
    for (name, m), color in zip(metrics.items(), colors):
        fpr, tpr, _ = roc_curve(y_cls, m['preds'])
        auc = roc_auc_score(y_cls, m['preds'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], '--', color='#404060', linewidth=1)
    ax.set_xlabel("False Positive Rate", color='#a0a0c0')
    ax.set_ylabel("True Positive Rate", color='#a0a0c0')
    ax.set_title("ROC — Popular vs. Not Popular", color='#ffffff', fontweight='bold')
    ax.tick_params(colors='#a0a0c0')
    ax.spines[:].set_color('#303050')
    ax.legend(facecolor='#1a1a2e', labelcolor='#c0c0e0', fontsize=10)
    st.pyplot(fig); plt.close()

    st.markdown('<p class="section-title">How the Hybrid Score Works</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="song-card">
        <div style="font-family:monospace;color:#a0c0ff;font-size:15px;text-align:center;padding:10px 0">
            score = α × (predicted popularity / 100) + (1 − α) × (1 − cosine distance)
        </div>
        <div class="song-meta" style="text-align:center;margin-top:10px">
            α = 0.7 by default &nbsp;·&nbsp; 70% popularity signal + 30% audio similarity
        </div>
    </div>
    """, unsafe_allow_html=True)
