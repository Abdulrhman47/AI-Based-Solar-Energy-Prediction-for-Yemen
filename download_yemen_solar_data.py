import urllib.request
import json
import csv
import time
import datetime

# Yemen cities: Latitude, Longitude
cities = {
    "Sanaa": (15.3694, 44.1910),
    "Aden": (12.7855, 45.0187),
    "Taiz": (13.5795, 44.0209),
    "Hodeidah": (14.7979, 42.9530),
    "Marib": (15.4625, 45.3258),
    "Ibb": (13.9667, 44.1833),
    "Mukalla": (14.5425, 49.1242),
    "Seiyun": (15.9430, 48.7873)
}

# Data period
start = "20150101"
end = "20251231"

# NASA POWER parameters
params = "ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M"

base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

rows = []
errors = []

print("Starting download from NASA POWER...")
print("--------------------------------------")

for city, (lat, lon) in cities.items():

    print("Downloading:", city)

    url = (
        f"{base_url}"
        f"?parameters={params}"
        f"&community=RE"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start}"
        f"&end={end}"
        f"&format=JSON"
    )

    try:

        with urllib.request.urlopen(url, timeout=120) as response:
            data = json.load(response)

        p = data["properties"]["parameter"]

        dates = sorted(
            p["ALLSKY_SFC_SW_DWN"].keys()
        )

        for d in dates:

            solar = p["ALLSKY_SFC_SW_DWN"].get(d, -999)
            temperature = p["T2M"].get(d, -999)
            humidity = p["RH2M"].get(d, -999)
            wind = p["WS2M"].get(d, -999)

            date = datetime.datetime.strptime(
                d,
                "%Y%m%d"
            ).date().isoformat()

            rows.append([
                date,
                city,
                lat,
                lon,
                solar,
                temperature,
                humidity,
                wind
            ])

        print("Completed:", city)

        time.sleep(1)

    except Exception as e:

        print("ERROR:", city)
        print(e)

        errors.append(
            (city, str(e))
        )


# CSV file name
out = "yemen_solar_nasa_power_2015_2025.csv"


with open(
    out,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Date",
        "City",
        "Latitude",
        "Longitude",
        "Solar_Radiation_kWh_m2_day",
        "Temperature_C",
        "Relative_Humidity_pct",
        "Wind_Speed_2m_m_s"
    ])

    writer.writerows(rows)


print()
print("===================================")
print("DATASET CREATED SUCCESSFULLY")
print("===================================")

print("File:", out)
print("Number of rows:", len(rows))
print("Number of cities:", len(cities))

if errors:

    print()
    print("Cities with errors:")

    for error in errors:
        print(error)

else:

    print("All cities downloaded successfully.")