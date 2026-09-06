import csv
import datetime
import math

# ==========================================
# Yemen Solar Dataset - Pre-processing
# ==========================================

input_file = "yemen_solar_nasa_power_2015_2025.csv"
output_file = "yemen_solar_cleaned.csv"

clean_rows = []

print("======================================")
print("YEMEN SOLAR DATA PRE-PROCESSING")
print("======================================")

# ------------------------------------------
# 1. Read original dataset
# ------------------------------------------

with open(input_file, "r", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    for row in reader:

        # Convert date
        date = datetime.datetime.strptime(
            row["Date"], "%Y-%m-%d"
        ).date()

        # ----------------------------------
        # Feature Engineering
        # ----------------------------------

        year = date.year
        month = date.month
        day = date.day

        day_of_year = date.timetuple().tm_yday

        # Cyclical month features
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)

        clean_rows.append({

            "Date": row["Date"],

            "City": row["City"],

            "Latitude": float(row["Latitude"]),

            "Longitude": float(row["Longitude"]),

            "Year": year,

            "Month": month,

            "Day": day,

            "DayOfYear": day_of_year,

            "Month_Sin": round(month_sin, 6),

            "Month_Cos": round(month_cos, 6),

            "Temperature_C":
                float(row["Temperature_C"]),

            "Relative_Humidity_pct":
                float(row["Relative_Humidity_pct"]),

            "Wind_Speed_2m_m_s":
                float(row["Wind_Speed_2m_m_s"]),

            # TARGET
            "Solar_Radiation_kWh_m2_day":
                float(row["Solar_Radiation_kWh_m2_day"])
        })


# ------------------------------------------
# 2. Save cleaned dataset
# ------------------------------------------

fieldnames = [

    "Date",
    "City",

    "Latitude",
    "Longitude",

    "Year",
    "Month",
    "Day",
    "DayOfYear",

    "Month_Sin",
    "Month_Cos",

    "Temperature_C",
    "Relative_Humidity_pct",
    "Wind_Speed_2m_m_s",

    "Solar_Radiation_kWh_m2_day"
]


with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(clean_rows)


print()
print("Pre-processing completed successfully!")

print("Original file:")
print(input_file)

print()

print("Cleaned file:")
print(output_file)

print()

print("Number of rows:", len(clean_rows))

print()

print("New Features:")
print("- Year")
print("- Month")
print("- Day")
print("- DayOfYear")
print("- Month_Sin")
print("- Month_Cos")

print()

print("Target:")
print("Solar_Radiation_kWh_m2_day")

print()

print("======================================")
print("CLEAN DATASET IS READY")
print("======================================")