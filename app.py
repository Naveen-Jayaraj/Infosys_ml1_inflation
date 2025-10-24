import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import calendar
import plotly.express as px
import plotly.graph_objects as go # Added for the advanced charts

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Cost of Living Index Forecaster", # NEW: Professional Name
    layout="wide",
    page_icon="📈" # NEW: Added a page icon
)

# ===============================
# Theme Toggle
# ===============================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

toggle = st.toggle("Dark Mode", value=(st.session_state.theme == "dark"))
st.session_state.theme = "dark" if toggle else "light"

if st.session_state.theme == "dark":
    # AMOLED Black Theme
    bg_gradient = "linear-gradient(135deg, #000000, #111111)" 
    text_color = "#EAEAEA" 
    card_bg = "#1C1C1E" 
    accent_color = "#FFA726" 
    border_color = "rgba(255, 255, 255, 0.15)"
    df_theme = "plotly_dark"
    # Glassmorphism Colors (Dark)
    glass_bg = "rgba(44, 44, 46, 0.6)" 
    glass_border = "rgba(255, 255, 255, 0.2)"
else:
    # Beige/Khaki Light Theme
    bg_gradient = "linear-gradient(135deg, #FAF3E0, #F5F5DC)" 
    text_color = "#333333" 
    card_bg = "#FFFFFF" 
    accent_color = "#4A6341" 
    border_color = "rgba(0,0,0,0.1)"
    df_theme = "plotly_white"
    # Glassmorphism Colors (Light)
    glass_bg = "rgba(255, 255, 255, 0.5)" 
    glass_border = "rgba(0, 0, 0, 0.15)"


# ===============================
# Custom CSS (MASSIVE UPDATE)
# ===============================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap');
.stApp {{
    background: {bg_gradient};
    color: {text_color};
    font-family: 'Lora', serif;
    transition: background 0.8s ease, color 0.4s ease;
}}

.main-title {{
    font-family: 'Playfair Display', serif !important;
    text-align: center;
    color: {text_color};
    font-size: 3rem;
    font-weight: 700;
    margin-top: -10px;
    margin-bottom: 0.5rem;
}}

.subtitle {{
    font-family: 'Lora', serif; 
    font-style: italic; 
    text-align: center;
    opacity: 0.85;
    font-size: 1.2rem;
    margin-bottom: 2.5rem;
}}

/* --- NEW: Widget Titles --- */
.widget-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    color: {accent_color};
    font-weight: 600;
    margin-bottom: -5px; /* Pulls widget closer */
}}

/* --- NEW: Glassmorphism Card Style --- */
.glass-card {{
    background: {glass_bg};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 25px;
    padding: 20px;
    border: 1px solid {glass_border};
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}
.glass-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.15);
}}

/* --- NEW: Style for Stat Cards (uses glass) --- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {glass_bg};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {glass_border};
    border-radius: 15px;
    padding: 10px 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMetric"] {{
    background-color: transparent;
    border: none;
    padding: 5px;
    text-align: center;
}}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMetric"] > label {{
    display: flex;
    justify-content: center;
    font-family: 'Lora', serif;
    color: {text_color};
    opacity: 0.8;
}}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    font-family: 'Lora', serif;
    color: {text_color};
}}

/* --- NEW: Custom Slider --- */
div[data-testid="stSlider"] {{
    padding-top: 5px;
}}
div[data-testid="stSlider"] > div[data-testid="stTickBar"] > div {{
    background: {accent_color}; /* Track color */
}}
div[data-testid="stSlider"] > div[data-testid="stThumb"] {{
    background: {accent_color}; /* Thumb color */
    border: 3px solid {card_bg};
    box-shadow: 0 0 10px {accent_color};
}}

/* --- NEW: Custom Tabs --- */
button[data-baseweb="tab"] {{
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: {text_color};
    opacity: 0.7;
    background: transparent;
    border-radius: 8px 8px 0 0;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    opacity: 1;
    color: {accent_color};
    background: {glass_bg}; /* Slight glass bg for active tab */
    border-bottom: 3px solid {accent_color};
}}
div[data-baseweb="tab-list"] {{
    border-bottom: 2px solid {border_color};
}}

/* --- NEW: Custom Multiselect --- */
div[data-testid="stMultiSelect"] {{
    border: 1px solid {border_color};
    border-radius: 8px;
    background: {glass_bg};
    backdrop-filter: blur(5px);
}}
div[data-testid="stMultiSelect"] > div {{
    border: none; /* Remove inner border */
    background: transparent;
}}
div[data-testid="stMultiSelect"] li {{ /* Dropdown items */
    font-family: 'Lora', serif;
}}

/* --- NEW: Custom Expander (for Definitions) --- */
div[data-testid="stExpander"] {{
    background: {glass_bg};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {glass_border};
    border-radius: 15px; /* Match stat cards */
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}
div[data-testid="stExpander"] > div[role="button"] > div {{
     font-family: 'Playfair Display', serif;
     font-size: 1.1rem;
     color: {text_color};
     font-weight: 600;
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
        # Note: These values should ideally be saved during training
        # Using placeholder values based on your code structure
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
# NEW: Updated Title and Subtitle
st.markdown('<h1 class="main-title">Cost of Living Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">An interactive dashboard for analyzing historical trends and forecasting future cost of living indices.</p>', unsafe_allow_html=True)

model, scaler = load_model_and_scaler()
df = load_dataset()

if model is None:
    st.warning("Model not found. Please ensure 'cost_of_living_model.keras' exists.")
elif df is None:
    st.warning("Dataset not found or invalid.")
else:
    st.markdown("---")
    
    # --- Inputs with Custom Titles ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<p class="widget-title">Select Year</p>', unsafe_allow_html=True)
        year = st.slider(
            "Select Year",
            min_value=int(df["Year"].min()),
            max_value=2027,
            value=min(2025, int(df["Year"].max()) + 1), # Default to next year
            step=1,
            label_visibility="collapsed" # Hide default label
        )
    with c2:
        st.markdown('<p class="widget-title">Select Month</p>', unsafe_allow_html=True)
        month = st.select_slider(
            "Select Month",
            options=[calendar.month_name[i] for i in range(1, 13)],
            value="January",
            label_visibility="collapsed" # Hide default label
        )
    month_num = list(calendar.month_name).index(month)

    # ===================================================================
    # --- Future Predictions (Vectorized for Speed) ---
    # ===================================================================
    cutoff_date = pd.to_datetime(f"{year}-{month_num}-01")
    last_real_year = int(df["Year"].max())
    
    future_df = pd.DataFrame() 

    if year > last_real_year:
        years_range = range(last_real_year + 1, year + 1)
        months_range = range(1, 13)
        index = pd.MultiIndex.from_product([years_range, months_range], names=["Year", "Month_Num"])
        future_dates_df = pd.DataFrame(index=index).reset_index()

        future_dates_df["Date"] = pd.to_datetime(
            future_dates_df["Year"].astype(str) + '-' + future_dates_df["Month_Num"].astype(str) + '-01'
        )
        future_dates_df = future_dates_df[future_dates_df["Date"] <= cutoff_date].copy()

        if not future_dates_df.empty:
            rural_inputs = future_dates_df[["Year", "Month_Num"]].copy()
            rural_inputs["Rural"] = 1.0
            rural_inputs["Urban"] = 0.0
            
            urban_inputs = future_dates_df[["Year", "Month_Num"]].copy()
            urban_inputs["Rural"] = 0.0
            urban_inputs["Urban"] = 1.0
            
            batch_input_np = pd.concat([rural_inputs, urban_inputs]).to_numpy(dtype='float32')
            
            scaled_inputs = scaler.transform(batch_input_np)
            all_predictions = model.predict(scaled_inputs, verbose=0)
            
            num_dates = len(future_dates_df)
            future_df = future_dates_df[["Year", "Month_Num", "Date"]].copy()
            future_df["Month"] = future_df["Month_Num"].apply(lambda m: calendar.month_name[m])
            future_df["Rural_Index"] = all_predictions[:num_dates].flatten()
            future_df["Urban_Index"] = all_predictions[num_dates:].flatten()

    df_extended = pd.concat([df, future_df], ignore_index=True)
    
    df_extended["Source"] = "Real"
    if year > last_real_year:
        df_extended.loc[df_extended["Year"] > last_real_year, "Source"] = "Predicted"

    filtered_df = df_extended[df_extended["Date"] <= cutoff_date]
    # ===================================================================
    # --- End of Vectorized Prediction Block ---
    # ===================================================================


    # --- Single-point Prediction for Gauges ---
    rural_input = np.array([[year, month_num, 1, 0]], dtype="float32")
    urban_input = np.array([[year, month_num, 0, 1]], dtype="float32")
    rural_pred = model.predict(scaler.transform(rural_input), verbose=0)[0][0]
    urban_pred = model.predict(scaler.transform(urban_input), verbose=0)[0][0]
    note = " (Predicted)" if year > last_real_year else ""

    st.markdown("---")

    # ===================================================================
    # --- NEW: Tabbed Output Section ---
    # ===================================================================
    tab1, tab2, tab3 = st.tabs([
        "📈 Key Predictions", 
        "📊 Trend Analysis", 
        "🧮 Summary Statistics"
    ])

    with tab1:
        # NEW: Added descriptive text
        st.markdown("Select a target year and month above to generate a real-time forecast. The gauges below visualize the predicted values for the 'Rural' and 'Urban' indices. A value of 100 represents the baseline cost from the start of the dataset.")
        st.markdown("---") # Visual separator
        
        # --- Advanced KPI Gauge Charts ---
        col1, col2 = st.columns(2)
        
        # --- Gauge 1: Rural Index ---
        with col1:
            fig_rural = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = rural_pred,
                title = {'text': f"Rural Index{note}", 'font': {'size': 22, 'family': 'Playfair Display', 'color': text_color}},
                number = {'font': {'size': 40, 'family': 'Lora', 'color': text_color}},
                gauge = {
                    'axis': {'range': [100, 250], 'tickwidth': 1, 'tickcolor': text_color},
                    'bar': {'color': accent_color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': border_color,
                    'steps': [
                        {'range': [100, 140], 'color': f"rgba({int(accent_color[1:3], 16)}, {int(accent_color[3:5], 16)}, {int(accent_color[5:7], 16)}, 0.2)"},
                        {'range': [140, 200], 'color': f"rgba({int(accent_color[1:3], 16)}, {int(accent_color[3:5], 16)}, {int(accent_color[5:7], 16)}, 0.5)"}
                    ],
                    'threshold': {
                        'line': {'color': "#e17055", 'width': 4},
                        'thickness': 0.75,
                        'value': 170
                    }
                }
            ))
            fig_rural.update_layout(
                paper_bgcolor = glass_bg,
                plot_bgcolor = 'rgba(0,0,0,0)',
                font = {'color': text_color, 'family': "Lora, serif"},
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                template=df_theme
            )
            # Add glass-card styling using a markdown wrapper
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.plotly_chart(fig_rural, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Gauge 2: Urban Index ---
        with col2:
            fig_urban = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = urban_pred,
                title = {'text': f"Urban Index{note}", 'font': {'size': 22, 'family': 'Playfair Display', 'color': text_color}},
                number = {'font': {'size': 40, 'family': 'Lora', 'color': text_color}},
                gauge = {
                    'axis': {'range': [100, 250], 'tickwidth': 1, 'tickcolor': text_color},
                    'bar': {'color': accent_color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': border_color,
                    'steps': [
                        {'range': [100, 140], 'color': f"rgba({int(accent_color[1:3], 16)}, {int(accent_color[3:5], 16)}, {int(accent_color[5:7], 16)}, 0.2)"},
                        {'range': [140, 200], 'color': f"rgba({int(accent_color[1:3], 16)}, {int(accent_color[3:5], 16)}, {int(accent_color[5:7], 16)}, 0.5)"}
                    ],
                    'threshold': {
                        'line': {'color': "#e17055", 'width': 4},
                        'thickness': 0.75,
                        'value': 170
                    }
                }
            ))
            fig_urban.update_layout(
                paper_bgcolor = glass_bg,
                plot_bgcolor = 'rgba(0,0,0,0)',
                font = {'color': text_color, 'family': "Lora, serif"},
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                template=df_theme
            )
            # Add glass-card styling using a markdown wrapper
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.plotly_chart(fig_urban, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        # NEW: Added descriptive text
        st.markdown("This tab visualizes the selected commodities over time. The **solid, filled line** represents historical data from the dataset, while the **dotted line** shows the predicted values generated by the model for future dates.")
        
        # ===================================================================
        # --- Trend Charts (Upgraded Visuals) ---
        # ===================================================================
        numeric_cols = [c for c in df_extended.select_dtypes(include=[np.number]).columns if c not in ["Year", "Month_Num"]]
        chart_cols = st.columns(3)
        plot_config = {"displayModeBar": True, "displaylogo": False, "scrollZoom": True, "responsive": True}

        # Define colors
        real_color = accent_color
        predicted_color = "#e17055" # A contrasting but nice orange
        grid_color = border_color # Use your theme's border color

        cols_to_plot = numeric_cols[:min(len(numeric_cols), 3)]

        for i, col in enumerate(cols_to_plot):
            df_real = filtered_df[filtered_df["Source"] == "Real"]
            df_pred = filtered_df[filtered_df["Source"] == "Predicted"]
            
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_real["Date"], 
                y=df_real[col],
                mode='lines',
                fill='tozeroy', 
                name='Real Data',
                line=dict(color=real_color, width=2.5),
                fillcolor=f"rgba({int(real_color[1:3], 16)}, {int(accent_color[3:5], 16)}, {int(accent_color[5:7], 16)}, 0.1)" 
            ))
            
            fig.add_trace(go.Scatter(
                x=df_pred["Date"], 
                y=df_pred[col],
                mode='lines+markers',
                name='Predicted Data',
                line=dict(color=predicted_color, dash='dot', width=3),
                marker=dict(size=6, symbol='circle')
            ))

            fig.update_layout(
                title=f"{col.replace('_', ' ')} Trend",
                title_font=dict(size=17, family="Lora, serif"), 
                template=df_theme,
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                xaxis=dict(gridcolor=grid_color, showline=True, linecolor=grid_color, zeroline=False),
                yaxis=dict(gridcolor=grid_color, showline=True, linecolor=grid_color, zeroline=False),
                hovermode="x unified", 
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(family="Lora, serif")
                ),
                font=dict(family="Lora, serif"), 
                margin=dict(l=5, r=5, t=50, b=10),
                height=400,
            )
            
            chart_cols[i % 3].plotly_chart(fig, use_container_width=True, config=plot_config)
        # ===================================================================
        # --- End of Trend Charts Block ---
        # =================================S==================================

    with tab3:
        # NEW: Added descriptive text
        st.markdown("Explore the statistical summary for commodities within your selected time frame (from the dataset's start up to your chosen date). Use the filter to add or remove items from the view.")
        
        # ===================================================================
        # --- Summary Stats (Card Layout with Filtering) ---
        # ===================================================================
        stats_df = filtered_df.select_dtypes(include=[np.number]).drop(
            columns=['Year', 'Month_Num'], errors='ignore'
        )
        
        if not stats_df.empty:
            all_commodities = stats_df.columns.tolist()
            
            default_selection = all_commodities[:min(len(all_commodities), 4)] 
            
            # --- NEW: Custom title for multiselect ---
            st.markdown('<p class="widget-title">Select commodities to view</p>', unsafe_allow_html=True)
            selected_commodities = st.multiselect(
                "Select commodities to view statistics:",
                options=all_commodities,
                default=default_selection,
                label_visibility="collapsed" # Hide default label
            )

            if not selected_commodities:
                st.info("Select one or more commodities above to display statistics.")
            else:
                stats_summary = stats_df[selected_commodities].agg(['min', 'max'])

                cols = st.columns(4) 
                
                for i, col_name in enumerate(selected_commodities):
                    with cols[i % 4]: 
                        st.markdown(
                            f"<h5 style='text-align: center; color: {accent_color}; font-family: \"Playfair Display\", serif;'>{col_name.replace('_', ' ')}</h5>", 
                            unsafe_allow_html=True
                        )
                        
                        min_val = stats_summary[col_name]['min']
                        max_val = stats_summary[col_name]['max']

                        with st.container(border=True):
                            st.metric(
                                label="Lowest",
                                value=f"{min_val:.2f}"
                            )
                            st.metric(
                                label="Highest",
                                value=f"{max_val:.2f}"
                            )
                        st.markdown("<br>", unsafe_allow_html=True)

        else:
            st.info("No data available for the selected period to display statistics.")

    # ===================================================================
    # --- NEW: Definitions Section at the bottom ---
    # ===================================================================
    st.markdown("---")
    st.markdown('<h2 class="main-title" style="font-size: 2rem; text-align: left; margin-top: 1.5rem;">About & Definitions</h2>', unsafe_allow_html=True)
    
    with st.expander("What is the 'Cost of Living Index (CLI)'?"):
        st.markdown("""
        The **Cost of Living Index** is a theoretical price index that measures the relative cost of maintaining a certain standard of living. It compares the expenses of a "basket" of goods and services (like food, housing, transport, and health) in one location or time period against another.
        
        * A **baseline of 100** is typically used for the starting point (e.g., a specific city or year).
        * An index of **120** means that it is 20% more expensive to live there than the baseline.
        * An index of **90** means it is 10% cheaper.
        """)

    with st.expander("About the 'Rural' & 'Urban' Indices"):
        st.markdown("""
        This dashboard forecasts two separate indices:
        1.  **Rural Index:** This index is calculated based on a basket of goods and services typically consumed by households in rural areas. It may give more weight to items like agricultural supplies, basic commodities, and different forms of transport.
        2.  **Urban Index:** This index is based on a consumption basket typical of city-dwelling households. It often places a higher emphasis on housing (rent), public transport, diverse food options, and recreation.
        
        The model predicts these two indices independently, as their costs can change at different rates.
        """)
        
    with st.expander("Forecasting Methodology"):
        st.markdown("""
        The predictive forecasts on this dashboard are generated by a **Deep Learning (Neural Network) model**. 
        
        The model was trained on the historical dataset of commodity prices, along with temporal features (Year, Month) and location features (Rural, Urban). It learned complex patterns and seasonal trends in the data to make its predictions for future dates. The values shown for future dates are the model's direct output.
        
        *(Note: This is a demonstration model. Data is illustrative.)*
        """)