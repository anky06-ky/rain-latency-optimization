from pathlib import Path

import pandas as pd
import requests


LATITUDE = 10.8231
LONGITUDE = 106.6297

START_DATE = "2021-10-01"
END_DATE = "2026-09-30"

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "weather_5years.csv"




def download_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
        ],
        "timezone": "Asia/Ho_Chi_Minh",
    }

    print("Downloading historical weather...")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(df.head())
    print("Rows:", len(df))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    download_weather()