"""
Download historical hourly weather observations for Tan Son Nhat Station (WMO 48900).

Data Source: Meteostat (https://meteostat.net/)
Station ID: 48900 (VVTS - Tan Son Nhat Airport, Ho Chi Minh City)
Coordinates: Lat 10.8167, Lon 106.6667, Alt 10m
License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
Citation: Meteostat Developers (2026). Hourly Weather Data for Station 48900.
"""

from datetime import datetime
from pathlib import Path
import warnings

# Monkeypatch for numpy 2.x compatibility with meteostat 1.6.5
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan

from meteostat import Hourly

# Filter deprecated pandas parse_dates warning emitted by legacy meteostat versions
warnings.filterwarnings("ignore", category=FutureWarning)

STATION_ID = "48900"
START_DATE = datetime(2021, 1, 1, 0, 0)
END_DATE = datetime(2026, 10, 29, 23, 59)
TIMEZONE = "Asia/Ho_Chi_Minh"

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "weather_meteostat_2021_2026.csv"


def download_meteostat_data():
    """Fetch hourly weather records from Meteostat API and save to CSV."""
    print("=" * 65)
    print(f"Downloading Meteostat observations for Station {STATION_ID}")
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} -> {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Timezone: {TIMEZONE}")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Query Meteostat hourly endpoint
    query = Hourly(STATION_ID, START_DATE, END_DATE, timezone=TIMEZONE)
    df = query.fetch()

    if df.empty:
        raise RuntimeError(f"No records retrieved for station {STATION_ID} in requested range.")

    # Reset index so 'time' becomes an explicit column
    df = df.reset_index()

    # Save to raw storage
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n[+] Download completed successfully!")
    print(f"    Saved to   : {OUTPUT_FILE}")
    print(f"    Rows       : {len(df):,}")
    print(f"    Time span  : {df['time'].min()} -> {df['time'].max()}")
    print("\n--- Column Overview & Missingness ---")
    missing = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    for col in df.columns:
        print(f"    {col:<8}: {len(df) - missing[col]:>6,} non-null | {missing[col]:>5,} missing ({missing_pct[col]:>5.2f}%)")

    return df


if __name__ == "__main__":
    download_meteostat_data()