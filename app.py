import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import calendar
import plotly.express as px

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Cost Predictor",
    layout="wide"
)

# ===============================
# Theme Toggle
# ===============================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

toggle = st.toggle("Dark Mode", value=(st.session_state.theme == "dark"))
st.session_state.theme = "dark" if toggle else "light"

if st.session_state.theme == "dark":
    bg_gradient = "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"
    text_color = "#f5f6fa"
    card_bg = "rgba(255, 255, 255, 0.05)"
    accent_color = "#00cec9"
    border_color = "rgba(255,255,255,0.15)"
    df_theme = "plotly_dark"
else:
    bg_gradient = "linear-gradient(135deg, #fdfbfb, #ebedee)"
    text_color = "#2d3436"
    card_bg = "rgba(255, 255, 255, 0.95)"
    accent_color = "#6c5ce7"
    border_color = "rgba(0,0,0,0.1)"
    df_theme = "plotly_white"

# ===============================
# Custom CSS
# ===============================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Playfair+Display:wght@600&display=swap');

.stApp {{
    background: {bg_gradient};
    color: {text_color};
    transition: background 0.8s ease, color 0.4s ease;
}}

.main-title {{
    font-family: 'Playfair Display', serif;
    text-align: center;
    color: {text_color};
    font-size: 3rem;
    margin-top: -10px;
    margin-bottom: 0.5rem;
}}

.subtitle {{
    font-family: 'Inter', sans-serif;
    text-align: center;
    opacity: 0.85;
    font-size: 1.2rem;
    margin-bottom: 2.5rem;
}}

.prediction-card {{
    background: {card_bg};
    border-radius: 25px;
    padding: 30px 25px;
    border: 1px solid {border_color};
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}
.prediction-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.2);
}}

.prediction-title {{
    font-weight: 600;
    font-size: 1.4rem;
    color: {accent_color};
    margin-bottom: 0.3rem;
}}

.prediction-value {{
    font-weight: 700;
    font-size: 2.8rem;
}}

.dataframe tbody tr:hover {{
    background-color: rgba(108, 92, 231, 0.05) !important;
}}
</style>
""", unsafe_allow_html=True)

# ===============================
# Load Model & Scaler
# ===============================
@st.cache_resource
def load_model_and_scaler():
    try:
        model = tf.keras.models.load_model("cost_of_living_model.keras")
        scaler = StandardScaler()
        scaler.mean_ = np.array([2016.5, 6.5, 0.5, 0.5], dtype='float32')
        scaler.scale_ = np.array([2.5, 3.45, 0.5, 0.5], dtype='float32')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("cost_of_living_data.csv")
        df.columns = [c.strip().title().replace(" ", "_") for c in df.columns]

        df["Year"] = (
            df["Year"].astype(str)
            .str.replace(".0", "", regex=False)
            .replace(["nan", "NaN", "None", ""], np.nan)
        )
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)

        def month_to_num(m):
            if pd.isna(m): return np.nan
            m = str(m).strip().title()
            if m.isdigit(): return int(m)
            try:
                return list(calendar.month_name).index(m)
            except ValueError:
                return np.nan

        df["Month_Num"] = df["Month"].apply(month_to_num)
        df = df.dropna(subset=["Month_Num"])
        df["Month_Num"] = df["Month_Num"].astype(int)
        df["Date"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["Month_Num"].astype(str) + "-01",
            errors="coerce"
        )
        return df
    except Exception as e:
        st.error(f"⚠️ Could not load dataset: {e}")
        return None

# ===============================
# App Layout
# ===============================
st.markdown('<h1 class="main-title">Cost Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional dashboard to visualize and predict cost-of-living trends</p>', unsafe_allow_html=True)

model, scaler = load_model_and_scaler()
df = load_dataset()

if model is None:
    st.warning("Model not found. Please ensure 'cost_of_living_model.keras' exists.")
elif df is None:
    st.warning("Dataset not found or invalid.")
else:
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        year = st.slider(
            "Select Year",
            min_value=int(df["Year"].min()),
            max_value=2100,
            value=min(2025, int(df["Year"].max())),
            step=1
        )
    with c2:
        month = st.select_slider(
            "Select Month",
            options=[calendar.month_name[i] for i in range(1, 13)],
            value="January"
        )
    month_num = list(calendar.month_name).index(month)

    # --- Future Predictions ---
    cutoff_date = pd.to_datetime(f"{year}-{month_num}-01")
    last_real_year = int(df["Year"].max())
    future_rows = []

    if year > last_real_year:
        for y in range(last_real_year + 1, year + 1):
            for m in range(1, 13):
                if y == year and m > month_num: break
                rural_input = np.array([[y, m, 1, 0]], dtype="float32")
                urban_input = np.array([[y, m, 0, 1]], dtype="float32")
                rural_pred = model.predict(scaler.transform(rural_input), verbose=0)[0][0]
                urban_pred = model.predict(scaler.transform(urban_input), verbose=0)[0][0]
                future_rows.append({
                    "Year": y, "Month_Num": m, "Month": calendar.month_name[m],
                    "Date": pd.to_datetime(f"{y}-{m}-01"),
                    "Rural_Index": rural_pred, "Urban_Index": urban_pred
                })
        future_df = pd.DataFrame(future_rows)
        df_extended = pd.concat([df, future_df], ignore_index=True)
    else:
        df_extended = df.copy()

    df_extended["Source"] = "Real"
    if year > last_real_year:
        df_extended.loc[df_extended["Year"] > last_real_year, "Source"] = "Predicted"

    filtered_df = df_extended[df_extended["Date"] <= cutoff_date]

    # --- Prediction Cards ---
    rural_input = np.array([[year, month_num, 1, 0]], dtype="float32")
    urban_input = np.array([[year, month_num, 0, 1]], dtype="float32")
    rural_pred = model.predict(scaler.transform(rural_input), verbose=0)[0][0]
    urban_pred = model.predict(scaler.transform(urban_input), verbose=0)[0][0]

    col1, col2 = st.columns(2)
    note = "<i style='color:#e17055'>(Predicted)</i>" if year > last_real_year else ""
    with col1:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-title">Rural Index</div>
            <div class="prediction-value">{rural_pred:.2f}</div>
            {note}
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-title">Urban Index</div>
            <div class="prediction-value">{urban_pred:.2f}</div>
            {note}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Trend Charts ---
    st.subheader("Trends (Historical & Predicted)")
    numeric_cols = [c for c in df_extended.select_dtypes(include=[np.number]).columns if c not in ["Year", "Month_Num"]]
    chart_cols = st.columns(3)
    plot_config = {"displayModeBar": True, "displaylogo": False, "scrollZoom": True, "responsive": True}

    for i, col in enumerate(numeric_cols[:3]):
        fig = px.line(
            filtered_df,
            x="Date", y=col, color="Source", line_dash="Source",
            markers=True, title=f"{col} Trend"
        )
        fig.update_layout(
            template=df_theme,
            title_font=dict(size=17),
            margin=dict(l=5, r=5, t=40, b=10),
            height=400,
            legend_title_text="Data Type"
        )
        chart_cols[i % 3].plotly_chart(fig, config=plot_config)

    # --- Summary Stats ---
    st.subheader("Summary Statistics")
    summary_df = filtered_df.describe().T
    st.dataframe(summary_df.style.background_gradient(cmap="Purples"), use_container_width=True)
