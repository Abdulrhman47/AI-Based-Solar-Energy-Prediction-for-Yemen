from pathlib import Path
import calendar
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Yemen Solar AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "yemen_solar_cleaned.csv"
MODEL_FILE = BASE / "best_solar_model.pkl"
RESULTS_FILE = BASE / "model_comparison_results.csv"

CITIES = {
    "Sanaa": (15.3694, 44.1910),
    "Aden": (12.7855, 45.0187),
    "Taiz": (13.5795, 44.0209),
    "Hodeidah": (14.7979, 42.9530),
    "Marib": (15.4625, 45.3258),
    "Ibb": (13.9667, 44.1833),
    "Mukalla": (14.5425, 49.1242),
    "Seiyun": (15.9430, 48.7873),
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data["Date"] = pd.to_datetime(data["Date"])
    return data

@st.cache_data
def load_results():
    if RESULTS_FILE.exists():
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame()

def suitability(r):
    if r >= 6.5: return "Excellent"
    if r >= 6.0: return "Very Good"
    if r >= 5.5: return "Good"
    if r >= 4.5: return "Moderate"
    return "Low"

def model_input(city, lat, lon, year, month, day, temp, humidity, wind):
    date = datetime(year, month, day)
    return pd.DataFrame({
        "City": [city],
        "Latitude": [lat],
        "Longitude": [lon],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "DayOfYear": [date.timetuple().tm_yday],
        "Month_Sin": [np.sin(2*np.pi*month/12)],
        "Month_Cos": [np.cos(2*np.pi*month/12)],
        "Temperature_C": [temp],
        "Relative_Humidity_pct": [humidity],
        "Wind_Speed_2m_m_s": [wind],
    })

def predict_one(model, city, lat, lon, year, month, day, temp, humidity, wind):
    return float(model.predict(model_input(
        city, lat, lon, year, month, day, temp, humidity, wind
    ))[0])

def monthly_forecast(model, city, lat, lon, year, temp, humidity, wind, size, efficiency):
    rows = []
    for m in range(1, 13):
        day = 15
        radiation = predict_one(model, city, lat, lon, year, m, day, temp, humidity, wind)
        days = calendar.monthrange(year, m)[1]
        daily_energy = radiation * size * efficiency
        rows.append({
            "Month": MONTH_NAMES[m-1],
            "Month_Number": m,
            "Solar_Radiation": radiation,
            "Monthly_Energy": daily_energy * days,
        })
    return pd.DataFrame(rows)

# ---------- Visual design ----------
st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
.hero {
  background: linear-gradient(135deg,#082f49 0%,#0e7490 55%,#16a34a 100%);
  padding: 28px 34px; border-radius: 22px; color:white; margin-bottom:18px;
  box-shadow: 0 12px 30px rgba(8,47,73,.18);
}
.hero h1 {margin:0; font-size:2.25rem;}
.hero p {margin:.45rem 0 0 0; opacity:.92; font-size:1.03rem;}
.kpi {
  background:white; border:1px solid #e2e8f0; border-radius:18px;
  padding:18px; box-shadow:0 6px 18px rgba(15,23,42,.06); min-height:118px;
}
.kpi .label {color:#64748b; font-size:.86rem; font-weight:700; text-transform:uppercase;}
.kpi .value {color:#0f172a; font-size:1.7rem; font-weight:800; margin-top:5px;}
.kpi .sub {color:#0e7490; font-size:.82rem; margin-top:4px;}
.section-title {font-size:1.35rem; font-weight:800; color:#0f172a; margin:.8rem 0 .4rem;}
[data-testid="stSidebar"] {background:#f8fafc;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>☀️ AI-Based Solar Energy Prediction System for Yemen</h1>
  <p>NASA POWER data • Machine Learning • City comparison • Energy forecasting • Interactive analytics</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_FILE.exists() or not DATA_FILE.exists():
    st.error("Required project files are missing. Keep best_solar_model.pkl and yemen_solar_cleaned.csv beside this app.")
    st.stop()

model = load_model()
df = load_data()
results = load_results()

with st.sidebar:
    st.header("Prediction Inputs")
    city = st.selectbox("City", list(CITIES.keys()))
    lat, lon = CITIES[city]
    st.caption(f"📍 {lat:.4f}, {lon:.4f}")

    year = st.number_input("Year", 2015, 2035, 2025, 1)
    c1, c2 = st.columns(2)
    month = c1.number_input("Month", 1, 12, 6, 1)
    day = c2.number_input("Day", 1, 31, 15, 1)

    temp = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0, 0.5)
    humidity = st.slider("Relative Humidity (%)", 0, 100, 30)
    wind = st.number_input("Wind Speed at 2m (m/s)", 0.0, 30.0, 3.0, 0.1)

    st.divider()
    size = st.number_input("Solar System Size (kW)", 0.1, 500.0, 5.0, 0.5)
    efficiency_pct = st.slider("System Efficiency (%)", 1, 100, 80)
    efficiency = efficiency_pct / 100.0

    run_prediction = st.button("⚡ Predict Solar Energy", use_container_width=True, type="primary")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard", "🔮 Prediction", "📊 EDA", "🤖 Model Performance"
])

# Calculate default/current prediction so dashboard is always alive.
safe_day = min(int(day), calendar.monthrange(int(year), int(month))[1])
radiation = predict_one(model, city, lat, lon, int(year), int(month), safe_day, temp, humidity, wind)
daily_energy = radiation * size * efficiency
days_in_month = calendar.monthrange(int(year), int(month))[1]
monthly_energy = daily_energy * days_in_month
forecast = monthly_forecast(model, city, lat, lon, int(year), temp, humidity, wind, size, efficiency)
best_row = forecast.loc[forecast["Solar_Radiation"].idxmax()]
annual_energy = forecast["Monthly_Energy"].sum()

with tab1:
    st.markdown('<div class="section-title">Executive Dashboard</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="kpi"><div class="label">Solar Radiation</div><div class="value">{radiation:.3f}</div><div class="sub">kWh/m²/day</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi"><div class="label">Suitability</div><div class="value">{suitability(radiation)}</div><div class="sub">{city}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi"><div class="label">Daily Energy</div><div class="value">{daily_energy:.2f}</div><div class="sub">kWh/day</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi"><div class="label">Annual Energy</div><div class="value">{annual_energy:,.0f}</div><div class="sub">kWh/year</div></div>', unsafe_allow_html=True)

    left,right = st.columns([1.5,1])
    with left:
        fig = px.line(
            forecast, x="Month", y="Solar_Radiation", markers=True,
            title=f"12-Month Solar Radiation Forecast — {city}"
        )
        fig.update_traces(line=dict(width=4), marker=dict(size=9))
        fig.update_layout(height=390, yaxis_title="kWh/m²/day", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Best Solar Month")
        st.metric(best_row["Month"], f'{best_row["Solar_Radiation"]:.3f} kWh/m²/day')
        st.metric("Monthly Energy", f'{best_row["Monthly_Energy"]:,.0f} kWh')
        st.info("Forecast uses the selected weather inputs while varying the month. It is a model-based scenario, not a weather forecast.")

    # ==============================
    # YEMEN SOLAR POTENTIAL MAP
    # ==============================

    st.subheader("☀️ Yemen Solar Potential Map")

    solar_map = (
        df.groupby("City", as_index=False)
        ["Solar_Radiation_kWh_m2_day"]
        .mean()
    )

    solar_map["Latitude"] = solar_map["City"].map(
        lambda x: CITIES[x][0]
    )

    solar_map["Longitude"] = solar_map["City"].map(
        lambda x: CITIES[x][1]
    )

    solar_map["Suitability"] = solar_map[
        "Solar_Radiation_kWh_m2_day"
    ].apply(suitability)

    solar_map["Solar Radiation"] = solar_map[
        "Solar_Radiation_kWh_m2_day"
    ].round(3)

    fig_map = px.scatter_map(
        solar_map,
        lat="Latitude",
        lon="Longitude",
        size="Solar Radiation",
        color="Solar Radiation",
        hover_name="City",
        hover_data={
            "Solar Radiation": ":.3f",
            "Suitability": True,
            "Latitude": ":.4f",
            "Longitude": ":.4f",
            "Solar_Radiation_kWh_m2_day": False,
        },
        size_max=35,
        zoom=4.8,
        center={"lat": 15.3, "lon": 47.0},
        color_continuous_scale="Turbo",
        labels={
            "Solar Radiation": "Solar Radiation (kWh/m²/day)"
        },
        title="Solar Energy Potential Across Yemen"
    )

    fig_map.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

with tab2:
    st.markdown('<div class="section-title">Interactive Prediction</div>', unsafe_allow_html=True)
    st.success(f"Predicted solar radiation for **{city}**: **{radiation:.3f} kWh/m²/day** — {suitability(radiation)}")
    c1,c2,c3 = st.columns(3)
    c1.metric("Expected Daily Energy", f"{daily_energy:.2f} kWh/day")
    c2.metric("Expected Monthly Energy", f"{monthly_energy:.2f} kWh/month")
    c3.metric("Estimated Annual Energy", f"{annual_energy:.2f} kWh/year")

    fig_energy = px.bar(
        forecast, x="Month", y="Monthly_Energy",
        title="Estimated Monthly PV Energy Production"
    )
    fig_energy.update_layout(height=390, yaxis_title="kWh/month", xaxis_title="")
    st.plotly_chart(fig_energy, use_container_width=True)

    st.dataframe(
        forecast[["Month","Solar_Radiation","Monthly_Energy"]]
        .rename(columns={"Solar_Radiation":"Solar Radiation (kWh/m²/day)",
                         "Monthly_Energy":"Energy (kWh/month)"}),
        use_container_width=True, hide_index=True
    )

with tab3:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    city_avg = df.groupby("City", as_index=False)["Solar_Radiation_kWh_m2_day"].mean()
    city_avg = city_avg.sort_values("Solar_Radiation_kWh_m2_day", ascending=False)
    f1 = px.bar(city_avg, x="City", y="Solar_Radiation_kWh_m2_day",
                title="Average Solar Radiation by City")
    f1.update_layout(yaxis_title="kWh/m²/day", xaxis_title="")
    st.plotly_chart(f1, use_container_width=True)

    monthly = df.groupby("Month", as_index=False)["Solar_Radiation_kWh_m2_day"].mean()
    monthly["Month_Name"] = monthly["Month"].map(lambda x: MONTH_NAMES[int(x)-1])
    f2 = px.line(monthly, x="Month_Name", y="Solar_Radiation_kWh_m2_day",
                 markers=True, title="Average Monthly Solar Radiation — Yemen")
    f2.update_layout(yaxis_title="kWh/m²/day", xaxis_title="")
    st.plotly_chart(f2, use_container_width=True)

    c1,c2 = st.columns(2)
    sample = df.sample(min(5000, len(df)), random_state=42)
    c1.plotly_chart(px.scatter(
        sample, x="Temperature_C", y="Solar_Radiation_kWh_m2_day",
        opacity=.35, title="Temperature vs Solar Radiation"
    ), use_container_width=True)
    c2.plotly_chart(px.scatter(
        sample, x="Relative_Humidity_pct", y="Solar_Radiation_kWh_m2_day",
        opacity=.35, title="Humidity vs Solar Radiation"
    ), use_container_width=True)

    numeric_cols = [
        "Latitude","Longitude","Month","DayOfYear","Temperature_C",
        "Relative_Humidity_pct","Wind_Speed_2m_m_s","Solar_Radiation_kWh_m2_day"
    ]
    corr = df[numeric_cols].corr()
    heat = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Correlation Heatmap"
    )
    heat.update_layout(height=620)
    st.plotly_chart(heat, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Machine Learning Performance</div>', unsafe_allow_html=True)

    if not results.empty:
        display = results.copy().sort_values("R2", ascending=False)
        best = display.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Best Model", best["Model"])
        c2.metric("R²", f'{best["R2"]:.4f}')
        c3.metric("MAE", f'{best["MAE"]:.4f}')
        c4.metric("RMSE", f'{best["RMSE"]:.4f}')

        comp = px.bar(display, x="Model", y="R2", text_auto=".3f",
                      title="R² Model Comparison")
        comp.update_layout(yaxis_title="R² Score", xaxis_title="")
        st.plotly_chart(comp, use_container_width=True)

        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.warning("model_comparison_results.csv was not found.")

    # Feature importance when the selected model supports it.
    try:
        estimator = model.named_steps["model"]
        pre = model.named_steps["preprocessor"]
        if hasattr(estimator, "feature_importances_"):
            names = pre.get_feature_names_out()
            imp = pd.DataFrame({
                "Feature": names,
                "Importance": estimator.feature_importances_
            }).sort_values("Importance", ascending=False).head(15)

            fig_imp = px.bar(
                imp.sort_values("Importance"),
                x="Importance", y="Feature", orientation="h",
                title="Top 15 Feature Importances"
            )
            fig_imp.update_layout(height=520)
            st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        pass

st.caption("Data source: NASA POWER | Academic machine-learning project for Yemen solar-energy analysis")
