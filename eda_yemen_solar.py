import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# Yemen Solar Energy - EDA
# ==========================================

file_name = "yemen_solar_cleaned.csv"
output_folder = "Yemen_Solar_EDA"

os.makedirs(output_folder, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(file_name)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

target = "Solar_Radiation_kWh_m2_day"

# ==========================================
# 1. Average Solar Radiation by City
# ==========================================

city_avg = (
    df.groupby("City")[target]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
city_avg.plot(kind="bar")

plt.title("Average Solar Radiation by City - Yemen")
plt.xlabel("City")
plt.ylabel("Solar Radiation (kWh/m2/day)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "01_solar_radiation_by_city.png"),
    dpi=300
)

plt.close()

print("Figure 1 completed.")


# ==========================================
# 2. Monthly Solar Radiation
# ==========================================

monthly_avg = (
    df.groupby("Month")[target]
    .mean()
)

plt.figure(figsize=(10, 6))
plt.plot(
    monthly_avg.index,
    monthly_avg.values,
    marker="o"
)

plt.title("Average Monthly Solar Radiation - Yemen")
plt.xlabel("Month")
plt.ylabel("Solar Radiation (kWh/m2/day)")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "02_monthly_solar_radiation.png"),
    dpi=300
)

plt.close()

print("Figure 2 completed.")


# ==========================================
# 3. Annual Solar Radiation
# ==========================================

yearly_avg = (
    df.groupby("Year")[target]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.plot(
    yearly_avg.index,
    yearly_avg.values,
    marker="o"
)

plt.title("Average Annual Solar Radiation - Yemen")
plt.xlabel("Year")
plt.ylabel("Solar Radiation (kWh/m2/day)")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "03_annual_solar_radiation.png"),
    dpi=300
)

plt.close()

print("Figure 3 completed.")


# ==========================================
# 4. Temperature vs Solar Radiation
# ==========================================

sample = df.sample(
    min(5000, len(df)),
    random_state=42
)

plt.figure(figsize=(10, 6))

plt.scatter(
    sample["Temperature_C"],
    sample[target],
    alpha=0.3
)

plt.title("Temperature vs Solar Radiation")
plt.xlabel("Temperature (C)")
plt.ylabel("Solar Radiation (kWh/m2/day)")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "04_temperature_vs_solar.png"),
    dpi=300
)

plt.close()

print("Figure 4 completed.")


# ==========================================
# 5. Humidity vs Solar Radiation
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    sample["Relative_Humidity_pct"],
    sample[target],
    alpha=0.3
)

plt.title("Relative Humidity vs Solar Radiation")
plt.xlabel("Relative Humidity (%)")
plt.ylabel("Solar Radiation (kWh/m2/day)")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "05_humidity_vs_solar.png"),
    dpi=300
)

plt.close()

print("Figure 5 completed.")


# ==========================================
# 6. Wind Speed vs Solar Radiation
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    sample["Wind_Speed_2m_m_s"],
    sample[target],
    alpha=0.3
)

plt.title("Wind Speed vs Solar Radiation")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Solar Radiation (kWh/m2/day)")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "06_wind_vs_solar.png"),
    dpi=300
)

plt.close()

print("Figure 6 completed.")


# ==========================================
# 7. Monthly Radiation for Every City
# ==========================================

city_month = df.groupby(
    ["City", "Month"]
)[target].mean().unstack(0)

plt.figure(figsize=(12, 7))

for city in city_month.columns:

    plt.plot(
        city_month.index,
        city_month[city],
        marker="o",
        label=city
    )

plt.title("Monthly Solar Radiation by City")
plt.xlabel("Month")
plt.ylabel("Solar Radiation (kWh/m2/day)")
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "07_monthly_radiation_all_cities.png"),
    dpi=300
)

plt.close()

print("Figure 7 completed.")


# ==========================================
# Summary
# ==========================================

print()
print("======================================")
print("EDA COMPLETED SUCCESSFULLY")
print("======================================")

print("Figures saved in:")
print(output_folder)

print()

print("Average Solar Radiation by City:")
print(city_avg)

print()

print("Best City:")
print(city_avg.index[0])

print("Average:")
print(round(city_avg.iloc[0], 3), "kWh/m2/day")

print()

print("Best Month:")
print(monthly_avg.idxmax())

print("Average:")
print(round(monthly_avg.max(), 3), "kWh/m2/day")