# ☀️ AI-Based Solar Energy Prediction System for Yemen

A Machine Learning project for predicting daily solar radiation and analyzing solar energy potential across major cities in Yemen.

The system uses historical solar and weather data from **NASA POWER**, performs data preprocessing and exploratory data analysis, trains multiple Machine Learning models, selects the best-performing model, and provides an interactive **Streamlit Dashboard** for prediction and visualization.

---

## 🎯 Project Objective

The main objective of this project is to build an AI-based system capable of predicting:

**Daily Solar Radiation (kWh/m²/day)**

The project also estimates solar energy production and helps analyze the solar potential of different Yemeni cities.

---

## 🌍 Study Area

The project covers 8 major cities in Yemen:

- Sanaa
- Aden
- Taiz
- Hodeidah
- Marib
- Ibb
- Mukalla
- Seiyun

---

## 📊 Data Source

The dataset is collected directly from:

**NASA POWER — Prediction Of Worldwide Energy Resources**

Study period:

**2015 – 2025**

The downloaded dataset contains more than **32,000 records**.

---

## 📌 Main Dataset Features

The project uses the following variables:

- Date
- City
- Latitude
- Longitude
- Solar Radiation
- Temperature
- Relative Humidity
- Wind Speed

The prediction target is:

`Solar_Radiation_kWh_m2_day`

---

## 🧹 Data Preprocessing

The preprocessing stage includes:

- Date conversion
- Data cleaning
- Missing-value checking
- Dataset preparation
- Feature engineering

Additional time-based features are created:

- Year
- Month
- Day
- DayOfYear
- Month_Sin
- Month_Cos

The processed dataset is saved as:

`yemen_solar_cleaned.csv`

---

## 📈 Exploratory Data Analysis — EDA

The project includes several visualizations to understand solar and weather patterns:

- Average solar radiation by city
- Monthly solar radiation
- Annual solar radiation
- Temperature vs Solar Radiation
- Humidity vs Solar Radiation
- Wind Speed vs Solar Radiation
- Monthly radiation comparison across cities
- Correlation analysis

---

## 🤖 Machine Learning Models

Three regression models are trained and compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The dataset is divided into:

- Training Data
- Testing Data

---

## 📏 Evaluation Metrics

The models are evaluated using:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² — Coefficient of Determination

### Best Model

The best-performing model in the current experiment is:

**Random Forest Regressor**

Results:

- **R² = 0.6359**
- **MAE = 0.3481**
- **RMSE = 0.5055**

---

## 🖥️ Interactive Streamlit Dashboard

The project includes an interactive web application developed using **Streamlit**.

Main sections:

### Dashboard
- Solar radiation prediction
- Solar energy estimation
- Monthly forecast
- Annual energy estimation
- Yemen solar potential map

### Prediction
Allows the user to enter weather, date, location, and solar-system parameters and obtain a solar radiation prediction.

### EDA
Displays interactive data visualizations and relationships between variables.

### Model Performance
Displays:

- Best Machine Learning model
- R²
- MAE
- RMSE
- Model comparison
- Feature importance

---

## 🗺️ Yemen Solar Potential Map

The Streamlit application contains an interactive geographical map showing the solar-energy potential of the selected Yemeni cities.

The map displays:

- City
- Solar radiation
- Latitude
- Longitude
- Solar suitability

---

## 🔄 AI / Machine Learning Pipeline

The complete project workflow is:

`NASA POWER Data`

↓

`Data Collection`

↓

`Data Preprocessing`

↓

`Exploratory Data Analysis`

↓

`Feature Engineering`

↓

`Train/Test Split`

↓

`Machine Learning Model Training`

↓

`Model Evaluation`

↓

`Best Model Selection`

↓

`Solar Radiation Prediction`

↓

`Streamlit Dashboard`

---

## 📁 Project Files

```text
AI-Based-Solar-Energy-Prediction-for-Yemen/
│
├── download_yemen_solar_data.py
├── preprocess_yemen_solar.py
├── eda_yemen_solar.py
├── train_yemen_solar_models.py
├── streamlit_yemen_solar.py
│
├── requirements_solar_dashboard.txt
│
├── yemen_solar_nasa_power_2015_2025.csv
├── yemen_solar_cleaned.csv
│
├── model_comparison_results.csv
├── model_comparison.png
│
└── README.md
