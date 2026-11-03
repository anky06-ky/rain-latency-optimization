"""
Module xây dựng đặc trưng (nguồn định nghĩa duy nhất cho đặc trưng, dùng chung cho offline và online).

- load_clean_weather_data: cổng kiểm soát chất lượng (2023-2025) + điền khuyết nhân quả nghiêm ngặt.
- build_features: tạo theo lô đặc trưng trễ / thống kê trượt / chu kỳ sin/cos cho cửa sổ L, mục tiêu y_{t+1}.
  Lưu ý: một số dòng bị loại (warm-up của đặc trưng trễ, thiếu mục tiêu) và kết quả được ĐÁNH LẠI CHỈ MỤC,
  nên chỉ mục không khớp với DataFrame sạch; hãy tra mốc thời gian qua cột 'time' / 'target_time' của nó.
- split_time_series: chia tập Train / Val / Test theo trình tự thời gian dựa trên target_time.
- extract_single_sample_features / extract_features_from_array: bộ trích xuất online cho một mẫu đơn,
  khớp từng bit với build_features (xem tests/test_pipeline.py).

Số đặc trưng: 6*L đặc trưng trễ + 5 thống kê trượt (chỉ khi L >= 3) + 4 mã hóa chu kỳ sin/cos
(L=1 -> 10 đặc trưng, L=6 -> 45 đặc trưng).
"""

import math
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("data/raw/weather_meteostat_2021_2026.csv")
PROCESSED_DATA_DIR = Path("data/processed")

# Các thông số khí tượng vật lý cơ bản
BASE_NUMERIC_COLS = ["temp", "dwpt", "rhum", "pres", "wspd"]
# Đầu vào thô theo từng giờ dùng để tạo đặc trưng trễ (thứ tự này quyết định thứ tự cột đặc trưng)
RAW_WINDOW_COLS = BASE_NUMERIC_COLS + ["prcp_clean"]
# Ngưỡng GO/NO-GO của cổng kiểm soát chất lượng: tỷ lệ thiếu lượng mưa của mỗi năm không vượt 5%
MAX_PRCP_MISSING_RATE = 0.05


def check_quality_gate(df: pd.DataFrame) -> pd.Series:
    """
    Cổng kiểm soát chất lượng: tính tỷ lệ thiếu `prcp` (chưa điền khuyết) theo từng năm và báo lỗi nếu có năm
    vượt MAX_PRCP_MISSING_RATE. Đây là quy tắc đã loại năm 2021 (thiếu 100%) và 2022 (40.23%) trong
    notebooks/01_weather_eda.ipynb; giai đoạn 2023-2025 thiếu lần lượt 0% / 2.93% / 3.47% (gộp 2.13%).
    Trả về: Series tỷ lệ thiếu theo năm.
    """
    rates = df["prcp"].isna().groupby(df["time"].dt.year).mean()
    failing = rates[rates > MAX_PRCP_MISSING_RATE]
    if not failing.empty:
        detail = ", ".join(f"{year}: {rate:.2%}" for year, rate in failing.items())
        raise ValueError(f"Quality gate failed: prcp missing rate above {MAX_PRCP_MISSING_RATE:.0%} ({detail})")
    return rates


def load_clean_weather_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Đọc file CSV thô của Meteostat và lọc qua cổng kiểm soát chất lượng (2023-2025, mỗi năm thiếu prcp ≤ 5%).
    Dùng điền khuyết nhân quả nghiêm ngặt (chỉ dùng quá khứ) để tránh rò rỉ dữ liệu tương lai:
    - Các biến vật lý liên tục (temp, dwpt, rhum, pres, wspd): điền tiến (ffill) tối đa 2 giờ.
    - Lượng mưa: điền tiến khoảng trống ngắn tối đa 1 giờ, phần còn thiếu coi là 0.0 khi tạo đặc trưng trễ.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}. Run 'python src/data/download.py' first.")

    df = pd.read_csv(filepath)
    df["time"] = pd.to_datetime(df["time"])

    # Cổng kiểm soát chất lượng: lọc giai đoạn dữ liệu sạch (2023-01-01 đến 2025-12-31)
    df = df[(df["time"] >= "2023-01-01") & (df["time"] <= "2025-12-31 23:59:59")].copy()
    df = df.sort_values("time").reset_index(drop=True)
    # Kiểm tra tỷ lệ thiếu lượng mưa theo năm trên dữ liệu gốc, trước khi điền khuyết
    check_quality_gate(df)

    # Điền tiến (ffill) nhân quả nghiêm ngặt cho các biến thời tiết liên tục (không dùng giá trị tương lai)
    df[BASE_NUMERIC_COLS] = df[BASE_NUMERIC_COLS].ffill(limit=2)

    # Làm sạch lượng mưa để tạo đặc trưng trễ: điền tiến chỉ dùng quá khứ (limit=1), phần còn lại gán 0.0
    # Lưu ý: giá trị mục tiêu thực tế lấy từ df['prcp'] thô trước fillna; mục tiêu bị thiếu sẽ bị loại bỏ
    df["prcp_clean"] = df["prcp"].ffill(limit=1).fillna(0.0)

    return df


def build_features(df: pd.DataFrame, L: int = 6) -> Tuple[pd.DataFrame, List[str]]:
    """
    Tạo các đặc trưng trễ, thống kê trượt và mã hóa chu kỳ sin/cos với độ dài cửa sổ lịch sử L cho trước.
    
    Tham số:
        df: DataFrame theo giờ đã làm sạch.
        L: Độ dài cửa sổ lịch sử tính theo giờ (L thuộc [1, 72]).
        
    Trả về:
        df_features: DataFrame chứa các đặc trưng, 'target' (y_t+1), 'time' và 'target_time'.
        feature_cols: Danh sách tên các cột đặc trưng.
    """
    if L < 1:
        raise ValueError("Window length L must be at least 1.")

    data_dict = {}

    # Mục tiêu: lượng mưa của 1 giờ tới (y_t+1)
    data_dict["target"] = df["prcp"].shift(-1)
    data_dict["time"] = df["time"]
    # Mốc thời gian mục tiêu tường minh để loại bỏ rò rỉ dữ liệu tại ranh giới chia tập
    data_dict["target_time"] = df["time"] + pd.to_timedelta(1, unit="h")

    feature_cols = []

    # 1. Đặc trưng trễ: giá trị từ t (lag 0) lùi về t-(L-1) (lag L-1)
    for lag in range(L):
        for col in RAW_WINDOW_COLS:
            col_name = f"{col}_lag_{lag}" if col != "prcp_clean" else f"prcp_lag_{lag}"
            data_dict[col_name] = df[col].shift(lag)
            feature_cols.append(col_name)

    # 2. Thống kê trượt trên cửa sổ L (khi L >= 3)
    if L >= 3:
        rolling_prcp = df["prcp_clean"].rolling(window=L)
        data_dict[f"prcp_rolling_mean_{L}"] = rolling_prcp.mean()
        data_dict[f"prcp_rolling_max_{L}"] = rolling_prcp.max()
        data_dict[f"prcp_rolling_std_{L}"] = rolling_prcp.std().fillna(0.0)

        data_dict[f"temp_rolling_mean_{L}"] = df["temp"].rolling(window=L).mean()
        data_dict[f"rhum_rolling_mean_{L}"] = df["rhum"].rolling(window=L).mean()

        feature_cols.extend([
            f"prcp_rolling_mean_{L}",
            f"prcp_rolling_max_{L}",
            f"prcp_rolling_std_{L}",
            f"temp_rolling_mean_{L}",
            f"rhum_rolling_mean_{L}",
        ])

    # 3. Mã hóa chu kỳ sin/cos theo thời gian (giờ trong ngày và tháng trong năm)
    hour = df["time"].dt.hour
    month = df["time"].dt.month
    data_dict["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    data_dict["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    data_dict["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    data_dict["cos_month"] = np.cos(2 * np.pi * month / 12.0)

    feature_cols.extend(["sin_hour", "cos_hour", "sin_month", "cos_month"])

    # Gộp thành DataFrame
    df_feat = pd.DataFrame(data_dict)

    # Bỏ mọi dòng có NaN ở đặc trưng (warm-up của đặc trưng trễ) hoặc NaN ở mục tiêu (thiếu giá trị thực tế)
    df_feat = df_feat.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    return df_feat, feature_cols


def split_time_series(df_feat: pd.DataFrame, feature_cols: List[str]) -> Tuple[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
]:
    """
    Chia chuỗi thời gian theo trình tự của target_time để loại bỏ rò rỉ dữ liệu tại ranh giới chia tập:
    - Tập Train: target_time <= 2024-12-31 23:59:59 (đặc trưng đến 22:00 để dự báo 23:00)
    - Tập Val: 2025-01-01 00:00:00 <= target_time <= 2025-06-30 23:59:59
    - Tập Test: 2025-07-01 00:00:00 <= target_time <= 2025-12-31 23:59:59
    """
    train_mask = (df_feat["target_time"] >= "2023-01-01 01:00:00") & (df_feat["target_time"] <= "2024-12-31 23:59:59")
    val_mask = (df_feat["target_time"] >= "2025-01-01 00:00:00") & (df_feat["target_time"] <= "2025-06-30 23:59:59")
    test_mask = (df_feat["target_time"] >= "2025-07-01 00:00:00") & (df_feat["target_time"] <= "2025-12-31 23:59:59")

    train_df = df_feat[train_mask]
    val_df = df_feat[val_mask]
    test_df = df_feat[test_mask]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def extract_features_from_array(arr: np.ndarray, hour: int, month: int, L: int) -> np.ndarray:
    """
    Bộ trích xuất đặc trưng cho một mẫu đơn (đường xử lý nhanh), làm việc trên bộ đệm NumPy thô.
    Đầu vào: mảng float kích thước (L, 6) với các cột RAW_WINDOW_COLS, dòng cũ nhất đứng đầu,
             cùng giờ/tháng của quan trắc mới nhất (thời điểm t).
    Đầu ra: mảng float32 2 chiều kích thước (1, num_features_D), thứ tự cột giống hệt build_features.
    """
    if arr.shape != (L, len(RAW_WINDOW_COLS)):
        raise ValueError(f"Expected window of shape ({L}, {len(RAW_WINDOW_COLS)}), got {arr.shape}")

    n_rolling = 5 if L >= 3 else 0
    out = np.empty((1, L * len(RAW_WINDOW_COLS) + n_rolling + 4), dtype=np.float32)

    # Các giá trị trễ từ lag 0 (mới nhất, dòng L-1) đến lag L-1 (cũ nhất, dòng 0)
    # Đảo thứ tự dòng bằng arr[::-1] rồi làm phẳng sẽ khớp chính xác thứ tự cột của build_features
    n_lag = L * len(RAW_WINDOW_COLS)
    out[0, :n_lag] = arr[::-1].ravel()

    # Thống kê trượt (khi L >= 3)
    if n_rolling:
        prcp_series = arr[:, 5]
        out[0, n_lag] = prcp_series.mean()
        out[0, n_lag + 1] = prcp_series.max()
        # Dùng ddof=1 để khớp chính xác với pandas rolling().std() trong build_features
        out[0, n_lag + 2] = prcp_series.std(ddof=1)
        out[0, n_lag + 3] = arr[:, 0].mean()
        out[0, n_lag + 4] = arr[:, 2].mean()

    # Mã hóa chu kỳ sin/cos từ mốc thời gian hiện tại (thời điểm t); math.* rẻ hơn np.* với số vô hướng
    out[0, -4] = math.sin(2.0 * math.pi * hour / 24.0)
    out[0, -3] = math.cos(2.0 * math.pi * hour / 24.0)
    out[0, -2] = math.sin(2.0 * math.pi * month / 12.0)
    out[0, -1] = math.cos(2.0 * math.pi * month / 12.0)

    return out


def extract_single_sample_features(raw_window_slice: pd.DataFrame, L: int) -> np.ndarray:
    """
    Bộ trích xuất đặc trưng vector hóa, dùng để đo độ trễ end-to-end trên một mẫu đơn.
    Đầu vào: DataFrame kích thước (L, num_features) biểu diễn L giờ gần nhất tính đến thời điểm t.
    Đầu ra: mảng NumPy 2 chiều kích thước (1, num_features_D).
    """
    # Lấy trực tiếp mảng dữ liệu nền của từng cột; chọn một tập con cột (df[[...]]) sẽ tạo DataFrame mới
    # ở mỗi lần gọi và từng chiếm phần lớn độ trễ khi xử lý một mẫu đơn.
    arr = np.column_stack([raw_window_slice[c].to_numpy(dtype=np.float32) for c in RAW_WINDOW_COLS])
    curr_time = raw_window_slice["time"].iloc[-1]
    return extract_features_from_array(arr, curr_time.hour, curr_time.month, L)


if __name__ == "__main__":
    print("=" * 65)
    print("Executing build_features.py standalone check...")
    print("=" * 65)

    clean_df = load_clean_weather_data()
    print(f"[+] Loaded clean weather data: {len(clean_df):,} rows.")

    # Minh họa với L = 3, 6, 24 và 72
    for test_L in [3, 6, 24, 72]:
        df_feat, fcols = build_features(clean_df, L=test_L)
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_time_series(df_feat, fcols)
        print(f"[*] Window L={test_L:>2}: Total Features={len(fcols):>3} | Train={X_tr.shape} | Val={X_v.shape} | Test={X_te.shape}")

    # Lưu tên các đặc trưng để tham khảo
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DATA_DIR / "feature_list.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(fcols))
    print(f"\n[+] Saved feature list ({len(fcols)} features for L=72) to {PROCESSED_DATA_DIR / 'feature_list.txt'}")
