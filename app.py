import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AgriPulse - Smart Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Unified, Crisp Professional Styling
# --------------------------------------------------
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Force Light Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

    /* Top Brand Bar */
    .top-header {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .top-header-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .top-header-desc {
        font-size: 13px;
        color: #64748b;
        margin-top: 4px;
        font-weight: 500;
    }
    .status-badge {
        background-color: #ecfdf5;
        color: #15803d;
        border: 1px solid #a7f3d0;
        font-size: 13px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
    }

    /* KPI Summary Cards */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #15803d;
        font-weight: 600;
        margin-top: 2px;
    }

    /* High Visibility Input Labels */
    label[data-testid="stWidgetLabel"] p {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        margin-bottom: 6px !important;
    }

    /* Text & Dropdown Input Boxes: Crisp Black Text on White */
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input,
    div[data-baseweb="select"] span {
        color: #0f172a !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }

    /* Primary Button: Clean Emerald */
    button[kind="primary"] {
        background-color: #15803d !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
    }
    button[kind="primary"]:hover {
        background-color: #166534 !important;
    }

    /* Result Card */
    .yield-box {
        background-color: #f0fdf4;
        border: 2px solid #86efac;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .yield-tag {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        color: #15803d;
        letter-spacing: 0.5px;
    }
    .yield-number {
        font-size: 38px;
        font-weight: 800;
        color: #14532d;
        margin: 6px 0;
    }
    .yield-unit {
        font-size: 20px;
        font-weight: 700;
        color: #15803d;
    }
    .yield-summary {
        color: #334155;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 8px;
    }

    /* Empty Placeholder */
    .idle-box {
        background-color: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 50px 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Cached Model & Dataset Loaders
# --------------------------------------------------
@st.cache_resource
def load_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "crop_yield_model.pkl")
    prep_path = os.path.join(base_dir, "models", "preprocessor.pkl")
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    return model, preprocessor

@st.cache_data
def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "crop_yield.csv")
    df = pd.read_csv(csv_path)
    df["Crop"] = df["Crop"].astype(str).str.strip()
    df["Season"] = df["Season"].astype(str).str.strip()
    df["State"] = df["State"].astype(str).str.strip()
    return df

try:
    model, preprocessor = load_pipeline()
    df = load_dataset()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    data_loaded = False
    df = pd.DataFrame()

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding-bottom: 14px; border-bottom: 1px solid #e2e8f0; margin-bottom: 16px;">
            <h2 style="margin: 0; font-size: 20px; font-weight: 800; color: #0f172a;">🌾 AgriPulse</h2>
            <div style="font-size: 12px; font-weight: 700; color: #15803d; letter-spacing: 0.5px;">PRECISION AGRI INTELLIGENCE</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Specifications")
    st.markdown("""
    - **Model:** Random Forest Regressor
    - **Validation:** 98.2% Accuracy (R²)
    - **Standard Unit:** **tonnes/hectare**
    """)

    st.markdown("---")
    st.markdown("### 💡 Yield Benchmarks")
    st.markdown("""
    - 🟢 **High:** `> 4.0 tonnes/hectare`
    - 🟡 **Moderate:** `2.0 - 4.0 tonnes/hectare`
    - 🔴 **Low:** `< 2.0 tonnes/hectare`
    """)


# --------------------------------------------------
# Top Header Banner
# --------------------------------------------------
st.markdown("""
<div class="top-header">
    <div>
        <h1 class="top-header-title">🌾 Crop Yield Prediction Dashboard</h1>
        <div class="top-header-desc">AI-powered harvest forecasting and crop intelligence.</div>
    </div>
    <div>
        <span class="status-badge">● Production Model Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# KPI Metrics Row
# --------------------------------------------------
if data_loaded and not df.empty:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Dataset Records</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-sub">Verified Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Supported Crops</div>
            <div class="kpi-value">{df['Crop'].nunique()}</div>
            <div class="kpi-sub">Field & Cash Crops</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Agricultural States</div>
            <div class="kpi-value">{df['State'].nunique()}</div>
            <div class="kpi-sub">Indian Regions</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Model Accuracy</div>
            <div class="kpi-value">98.2%</div>
            <div class="kpi-sub">R² Precision</div>
        </div>
        """, unsafe_allow_html=True)

st.write("") # Spacer


# --------------------------------------------------
# Main Navigation Tabs
# --------------------------------------------------
tab_pred, tab_analytics, tab_guide = st.tabs([
    "🔮 Yield Prediction",
    "📈 Agricultural Analytics",
    "📖 Model Details"
])


# --------------------------------------------------
# TAB 1: YIELD PREDICTION
# --------------------------------------------------
with tab_pred:
    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("<h3 style='color: #0f172a; font-size: 18px; font-weight: 700; margin: 0 0 14px 0;'>🌱 Input Parameters</h3>", unsafe_allow_html=True)

        if data_loaded:
            crop_list = sorted(df["Crop"].unique())
            season_list = sorted(df["Season"].unique())
            state_list = sorted(df["State"].unique())
        else:
            crop_list, season_list, state_list = ["Rice", "Wheat"], ["Kharif", "Rabi"], ["Punjab"]

        row1_1, row1_2 = st.columns(2)
        with row1_1:
            selected_crop = st.selectbox("🌾 Crop Variety", crop_list, index=crop_list.index("Rice") if "Rice" in crop_list else 0)
            selected_state = st.selectbox("📍 State / Region", state_list, index=0)
        with row1_2:
            selected_season = st.selectbox("☀️ Cultivation Season", season_list, index=0)
            crop_year = st.number_input("📅 Crop Year", min_value=1997, max_value=2030, value=2024, step=1)

        st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

        row2_1, row2_2 = st.columns(2)
        with row2_1:
            area_val = st.number_input("📐 Cultivation Area (Hectares)", min_value=0.1, value=2500.0, step=50.0)
            rainfall_val = st.number_input("🌧️ Annual Rainfall (mm)", min_value=0.0, value=1150.0, step=25.0)
        with row2_2:
            fertilizer_val = st.number_input("🧪 Fertilizer Used (kg)", min_value=0.0, value=350000.0, step=5000.0)
            pesticide_val = st.number_input("🧴 Pesticide Used (kg)", min_value=0.0, value=1800.0, step=50.0)

        st.write("")
        submit = st.button("🚀 Calculate Yield & Production", type="primary", use_container_width=True)

    with col_output:
        st.markdown("<h3 style='color: #0f172a; font-size: 18px; font-weight: 700; margin: 0 0 14px 0;'>📊 Prediction Results</h3>", unsafe_allow_html=True)

        if submit and data_loaded:
            with st.spinner("Calculating forecast..."):
                input_df = pd.DataFrame([{
                    "Crop": selected_crop,
                    "Crop_Year": crop_year,
                    "Season": selected_season,
                    "State": selected_state,
                    "Area": area_val,
                    "Annual_Rainfall": rainfall_val,
                    "Fertilizer": fertilizer_val,
                    "Pesticide": pesticide_val
                }])

                transformed = preprocessor.transform(input_df)
                pred_yield = round(float(model.predict(transformed)[0]), 2)
                est_production = round(pred_yield * area_val, 1)

                if pred_yield > 4.0:
                    status_badge = "🌟 High Productivity Zone"
                elif pred_yield >= 2.0:
                    status_badge = "🌾 Moderate Productivity Zone"
                else:
                    status_badge = "⚠️ Low Productivity Alert"

                # Store history
                st.session_state.history.append({
                    "Timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Crop": selected_crop,
                    "State": selected_state,
                    "Season": selected_season,
                    "Area (ha)": area_val,
                    "Yield": f"{pred_yield:.2f} tonnes/hectare",
                    "Est Production": f"{est_production:,.1f} tonnes"
                })

                # Result card
                st.markdown(f"""
                <div class="yield-box">
                    <div class="yield-tag">Estimated Yield Rate</div>
                    <div class="yield-number">
                        {pred_yield:.2f} <span class="yield-unit">tonnes/hectare</span>
                    </div>
                    <div style="margin: 8px 0;">
                        <span class="status-badge">{status_badge}</span>
                    </div>
                    <div class="yield-summary">
                        For <b>{area_val:,.0f} hectares</b> of <b>{selected_crop}</b> in <b>{selected_state} ({selected_season})</b>, 
                        the expected total output is <b>{est_production:,.1f} Metric Tonnes</b>.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge indicator
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_yield,
                    number={'suffix': " t/ha", 'font': {'size': 28, 'color': '#0f172a'}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Productivity Gauge ({pred_yield:.2f} tonnes/hectare)", 'font': {'size': 13, 'color': '#475569'}},
                    gauge={
                        'axis': {'range': [0, max(8, pred_yield * 1.3)], 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#15803d"},
                        'bgcolor': "#f1f5f9",
                        'steps': [
                            {'range': [0, 2], 'color': '#fee2e2'},
                            {'range': [2, 4], 'color': '#fef3c7'},
                            {'range': [4, 10], 'color': '#dcfce7'}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=230, margin=dict(l=15, r=15, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.markdown("""
            <div class="idle-box">
                <div style="font-size: 42px; margin-bottom: 8px;">🚜</div>
                <h4 style="color: #0f172a; margin: 0 0 4px 0; font-weight: 700;">Ready for Estimation</h4>
                <p style="color: #64748b; font-size: 13px; max-width: 320px; margin: 0 auto;">
                    Adjust the parameters on the left and click <b>Calculate Yield & Production</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Prediction History Table
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0f172a; font-weight: 700; margin-bottom: 8px;'>📋 Prediction History</h4>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear Log"):
            st.session_state.history = []
            st.rerun()


# --------------------------------------------------
# TAB 2: ANALYTICS
# --------------------------------------------------
with tab_analytics:
    st.markdown("<h3 style='color: #0f172a; font-size: 18px; font-weight: 700; margin-bottom: 6px;'>📈 Interactive Crop & Regional Analytics</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 13px; margin-bottom: 16px;'>Explore performance across crops, seasonal comparisons, and environmental impacts.</p>", unsafe_allow_html=True)

    if data_loaded and not df.empty:
        # Interactive Controls
        f1, f2 = st.columns([1, 1])
        with f1:
            crop_filter = st.selectbox(
                "🔍 Select Crop to Inspect",
                ["All Crops (Excl. Coconut Outlier)"] + sorted(df["Crop"].unique()),
                index=0,
                key="analytics_crop_select"
            )
        with f2:
            season_filter = st.multiselect(
                "☀️ Filter Seasons",
                options=sorted(df["Season"].unique()),
                default=sorted(df["Season"].unique()),
                key="analytics_season_filter"
            )

        # Prepare filtered dataset
        plot_df = df.copy()
        if season_filter:
            plot_df = plot_df[plot_df["Season"].isin(season_filter)]

        if crop_filter == "All Crops (Excl. Coconut Outlier)":
            plot_df = plot_df[plot_df["Crop"] != "Coconut"]
            view_label = "All Crops (Excl. Coconut)"
        else:
            plot_df = plot_df[plot_df["Crop"] == crop_filter]
            view_label = crop_filter

        g1, g2 = st.columns(2)
        with g1:
            st.markdown(f"<b style='color:#0f172a;'>Top Performing Crops by Average Yield (tonnes/ha)</b>", unsafe_allow_html=True)
            # Exclude coconut from general top-crops bar chart because coconut is counted in thousands of nuts/ha
            top_crops_df = df[df["Crop"] != "Coconut"].groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(10).reset_index()
            fig1 = px.bar(
                top_crops_df,
                x="Yield",
                y="Crop",
                orientation="h",
                color="Yield",
                color_continuous_scale="Greens",
                labels={"Yield": "Mean Yield (tonnes/ha)"}
            )
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)

        with g2:
            st.markdown(f"<b style='color:#0f172a;'>Average Yield by Cultivation Season: {view_label}</b>", unsafe_allow_html=True)
            season_avg = plot_df.groupby("Season")["Yield"].mean().reset_index()
            fig2 = px.bar(
                season_avg,
                x="Season",
                y="Yield",
                color="Season",
                text_auto=".2f",
                color_discrete_sequence=["#15803d", "#0284c7", "#f59e0b", "#7c3aed", "#ec4899", "#0d9488"],
                labels={"Yield": "Average Yield (tonnes/ha)"}
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"<br><b style='color:#0f172a;'>Rainfall vs. Yield Correlation: {view_label}</b>", unsafe_allow_html=True)
        scatter_sample = plot_df.sample(min(600, len(plot_df)), random_state=42) if len(plot_df) > 0 else plot_df
        fig3 = px.scatter(
            scatter_sample,
            x="Annual_Rainfall",
            y="Yield",
            size="Area",
            color="Crop",
            hover_name="State",
            size_max=24,
            opacity=0.75,
            labels={"Annual_Rainfall": "Annual Rainfall (mm)", "Yield": "Yield (tonnes/ha)"}
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------
# TAB 3: MODEL DETAILS
# --------------------------------------------------
with tab_guide:
    st.markdown("""
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px;">
        <h4 style="margin-top: 0; color: #0f172a;">🌾 Machine Learning Specifications</h4>
        <p style="color: #475569; font-size: 14px; line-height: 1.6;">
            AgriPulse uses a Random Forest Regressor trained on Indian agricultural data to predict crop yield per hectare.
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px;">
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0;">
                <b style="color: #0f172a;">Input Features:</b>
                <ul style="color: #475569; font-size: 13px; margin: 6px 0 0 0; padding-left: 18px; line-height: 1.8;">
                    <li><b>Crop & Season:</b> Crop variety and sowing cycle</li>
                    <li><b>State:</b> Geographical region</li>
                    <li><b>Cultivation Area:</b> In Hectares</li>
                    <li><b>Annual Rainfall:</b> In millimeters (mm)</li>
                    <li><b>Fertilizer & Pesticide:</b> Total applied (kg)</li>
                </ul>
            </div>
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0;">
                <b style="color: #0f172a;">Model Performance:</b>
                <ul style="color: #475569; font-size: 13px; margin: 6px 0 0 0; padding-left: 18px; line-height: 1.8;">
                    <li><b>Model:</b> Random Forest Regressor</li>
                    <li><b>Pre-processing:</b> One-Hot Encoding</li>
                    <li><b>R² Score:</b> ~0.98</li>
                    <li><b>Standard Metric:</b> <b>tonnes/hectare</b></li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 12px; margin-top: 30px; padding: 16px 0; border-top: 1px solid #e2e8f0;">
    AgriPulse • Precision Agriculture & Crop Yield Intelligence
</div>
""", unsafe_allow_html=True)
