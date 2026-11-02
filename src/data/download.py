"""
Tải dữ liệu quan trắc thời tiết lịch sử theo giờ của trạm Tân Sơn Nhất (WMO 48900).

Nguồn dữ liệu: Meteostat (https://meteostat.net/)
Mã trạm: 48900 (VVTS - Sân bay Tân Sơn Nhất, TP. Hồ Chí Minh)
Tọa độ: Vĩ độ 10.8167, Kinh độ 106.6667, Độ cao 10m
Giấy phép: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
Trích dẫn: Meteostat Developers (2026). Hourly Weather Data for Station 48900.
           Truy xuất ngày 2026-10-29 bằng meteostat 1.6.5; file dùng cho các kết quả đã báo cáo chứa
           45,944 bản ghi theo giờ từ 2021-01-01 00:00 đến 2026-03-30 07:00 (Asia/Ho_Chi_Minh).
"""

from datetime import datetime
from pathlib import Path
import warnings

# Monkeypatch giúp meteostat 1.6.5 tương thích với NumPy 2.x
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan

from meteostat import Hourly

# Lọc bỏ cảnh báo lỗi thời (deprecated) về parse_dates của pandas do các phiên bản meteostat cũ phát ra
warnings.filterwarnings("ignore", category=FutureWarning)

STATION_ID = "48900"
START_DATE = datetime(2021, 1, 1, 0, 0)
END_DATE = datetime(2026, 10, 29, 23, 59)
TIMEZONE = "Asia/Ho_Chi_Minh"

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "weather_meteostat_2021_2026.csv"


def download_meteostat_data():
    """Lấy dữ liệu thời tiết theo giờ từ API của Meteostat và lưu ra file CSV."""
    print("=" * 65)
    print(f"Downloading Meteostat observations for Station {STATION_ID}")
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} -> {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Timezone: {TIMEZONE}")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Truy vấn endpoint dữ liệu theo giờ của Meteostat
    query = Hourly(STATION_ID, START_DATE, END_DATE, timezone=TIMEZONE)
    df = query.fetch()

    if df.empty:
        raise RuntimeError(f"No records retrieved for station {STATION_ID} in requested range.")

    # Đặt lại chỉ mục để 'time' trở thành một cột tường minh
    df = df.reset_index()

    # Lưu vào thư mục dữ liệu thô
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