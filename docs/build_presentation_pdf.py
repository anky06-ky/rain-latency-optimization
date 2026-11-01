"""
Build a publication-quality Presentation Script & Defense PDF for the Rainfall Latency Optimization Project.
Compiles to PDF via Microsoft Edge headless with base64 embedded figures.
"""

import sys
import base64
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_PDF = DOCS_DIR / "KICH_BAN_THUYET_TRINH_BAO_CAO.pdf"
HTML_FILE = DOCS_DIR / "presentation_temp.html"

def img_to_b64(img_path: Path) -> str:
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

print("[*] Encoding figures to base64...")
b64_pareto = img_to_b64(FIGURES_DIR / "pareto_frontier.png")
b64_convergence = img_to_b64(FIGURES_DIR / "convergence_comparison.png")
b64_split = img_to_b64(FIGURES_DIR / "train_val_test_split.png")
b64_actual = img_to_b64(FIGURES_DIR / "actual_vs_predicted_test.png")
b64_missing = img_to_b64(FIGURES_DIR / "missing_rate_by_year.png")

html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Kịch bản Thuyết trình Báo cáo Đồ án - Tối ưu hóa Độ trễ Dự báo Lượng mưa</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  @page {
    size: A4;
    margin: 14mm 14mm 16mm 14mm;
    @bottom-right {
      content: "Trang " counter(page);
      font-size: 9px;
      color: #64748b;
      font-family: 'Inter', sans-serif;
    }
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1e293b;
    background: #ffffff;
    font-size: 11.5px;
    line-height: 1.55;
  }

  .cover-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
    color: #ffffff;
    padding: 24px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.12);
  }

  .badge-row {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }

  .badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 9999px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  .badge-blue { background: #0284c7; color: #ffffff; }
  .badge-emerald { background: #10b981; color: #ffffff; }
  .badge-amber { background: #f59e0b; color: #ffffff; }
  .badge-dark { background: rgba(255,255,255,0.15); color: #e2e8f0; border: 1px solid rgba(255,255,255,0.25); }

  h1.main-title {
    font-size: 19px;
    font-weight: 800;
    line-height: 1.3;
    margin-bottom: 8px;
    color: #ffffff;
  }

  .sub-title {
    font-size: 11.5px;
    color: #93c5fd;
    font-weight: 500;
    margin-bottom: 16px;
  }

  .meta-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.15);
    font-size: 10px;
  }

  .meta-item strong {
    display: block;
    color: #cbd5e1;
    font-size: 9px;
    text-transform: uppercase;
    margin-bottom: 2px;
  }

  .meta-item span {
    color: #ffffff;
    font-weight: 600;
  }

  /* Table styling */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 16px 0;
    font-size: 10.5px;
  }

  th, td {
    padding: 6px 10px;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
  }

  th {
    background-color: #f1f5f9;
    color: #334155;
    font-weight: 700;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  tr:nth-child(even) {
    background-color: #f8fafc;
  }

  .slide-container {
    page-break-inside: avoid;
    break-inside: avoid;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    margin-bottom: 18px;
    overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  }

  .slide-header {
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    padding: 10px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .slide-num {
    font-size: 12px;
    font-weight: 800;
    color: #0369a1;
  }

  .slide-time {
    font-size: 10px;
    font-weight: 600;
    color: #475569;
    background: #e2e8f0;
    padding: 2px 8px;
    border-radius: 9999px;
  }

  .slide-body {
    padding: 14px 16px;
  }

  .slide-bullets {
    margin-bottom: 12px;
  }

  .slide-bullets ul {
    padding-left: 18px;
    margin-top: 4px;
  }

  .slide-bullets li {
    margin-bottom: 4px;
    color: #334155;
  }

  .speaker-box {
    background: #f0fdf4;
    border-left: 4px solid #10b981;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    margin-top: 10px;
  }

  .speaker-header {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    font-weight: 700;
    color: #047857;
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  .speaker-script {
    font-size: 11px;
    font-style: italic;
    color: #1e293b;
    line-height: 1.5;
  }

  .speaker-script strong {
    font-style: normal;
    color: #0f172a;
    font-weight: 700;
  }

  .img-box {
    text-align: center;
    margin: 8px 0 10px 0;
  }

  .img-box img {
    max-width: 92%;
    max-height: 170px;
    height: auto;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  }

  .caption {
    font-size: 9.5px;
    color: #64748b;
    margin-top: 4px;
    font-weight: 500;
  }

  .alert-box {
    background: #fff1f2;
    border-left: 4px solid #e11d48;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 10px;
    color: #881337;
    margin-top: 8px;
  }

  .highlight-optuna {
    background-color: #dcfce7;
    font-weight: 700;
    color: #166534;
  }

  .faq-container {
    page-break-inside: avoid;
    break-inside: avoid;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 12px;
  }

  .faq-q {
    font-size: 11.5px;
    font-weight: 700;
    color: #b91c1c;
    margin-bottom: 6px;
    display: flex;
    gap: 6px;
  }

  .faq-a {
    font-size: 10.8px;
    color: #1e293b;
    line-height: 1.5;
    background: #f8fafc;
    padding: 8px 12px;
    border-left: 3px solid #0284c7;
    border-radius: 0 6px 6px 0;
  }

  .section-divider {
    page-break-before: always;
    break-before: page;
  }

  .section-header {
    font-size: 15px;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2px solid #0284c7;
    padding-bottom: 4px;
    margin: 16px 0 12px 0;
  }

  .key-takeaway-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 12px 0;
  }

  .takeaway-card {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 10px;
  }

  .takeaway-card h4 {
    font-size: 10.5px;
    color: #0369a1;
    margin-bottom: 4px;
  }
</style>
</head>
<body>

<!-- COVER / INTRO CARD -->
<div class="cover-card">
  <div class="badge-row">
    <span class="badge badge-blue">Đồ Án Khoa Học Dữ Liệu</span>
    <span class="badge badge-emerald">Tối Ưu Hóa Có Ràng Buộc</span>
    <span class="badge badge-amber">Thời Gian Thực Edge ML</span>
    <span class="badge badge-dark">Python 3.13 | LightGBM | Optuna</span>
  </div>
  <h1 class="main-title">KỊCH BẢN THUYẾT TRÌNH BÁO CÁO & BẢO VỆ ĐỒ ÁN</h1>
  <div class="sub-title">Tối ưu hóa Siêu tham số & Chiều sâu đặc trưng có Ràng buộc Độ trễ cho Dự báo Lượng mưa Ngắn hạn tại Trạm Tân Sơn Nhất</div>

  <div class="meta-grid">
    <div class="meta-item">
      <strong>Thời lượng báo cáo:</strong>
      <span>12 – 15 Phút (+ 5 Phút Q&A)</span>
    </div>
    <div class="meta-item">
      <strong>Ràng buộc kỹ thuật:</strong>
      <span>T_e2e ≤ 5.0 ms (Biên an toàn ≤ 4.5 ms)</span>
    </div>
    <div class="meta-item">
      <strong>Địa bàn nghiên cứu:</strong>
      <span>Trạm Tân Sơn Nhất (WMO 48900)</span>
    </div>
    <div class="meta-item">
      <strong>Tài nguyên dự án:</strong>
      <span>GitHub Repo (CI Green 100%)</span>
    </div>
  </div>
</div>

<!-- TIME ALLOCATION AGENDA -->
<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">LỊCH TRÌNH BÁO CÁO (15 PHÚT AGENDA)</span>
    <span class="slide-time">Tổng 12 Slide</span>
  </div>
  <div class="slide-body">
    <table>
      <thead>
        <tr>
          <th>Phân đoạn</th>
          <th>Nội dung trọng tâm</th>
          <th>Thời lượng</th>
          <th>Slide tương ứng</th>
          <th>Mục tiêu truyền tải</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Phần 1: Đặt vấn đề</strong></td>
          <td>Bối cảnh ngập lụt TP.HCM & Thách thức trễ thời gian thực</td>
          <td>2.5 phút</td>
          <td>Slide 1 – 2</td>
          <td>Khơi gợi nhu cầu cấp bách của SLA ≤ 5.0 ms trên thiết bị biên.</td>
        </tr>
        <tr>
          <td><strong>Phần 2: Phương pháp</strong></td>
          <td>Data Quality Gate, Chống rò rỉ & Phát biểu tối ưu có ràng buộc</td>
          <td>3.0 phút</td>
          <td>Slide 3 – 4</td>
          <td>Minh bạch dữ liệu, loại 2021-2022 bằng số liệu thực nghiệm.</td>
        </tr>
        <tr>
          <td><strong>Phần 3: Đối sánh</strong></td>
          <td>Giao thức công bằng: Optuna TPE vs Random Search (5 seeds, 500 trials)</td>
          <td>2.5 phút</td>
          <td>Slide 5</td>
          <td>Khẳng định tính khoa học: cùng không gian, cùng ngân sách.</td>
        </tr>
        <tr>
          <td><strong>Phần 4: Kết quả</strong></td>
          <td>Tỷ lệ khả thi vượt trội (+14 pp), Kiến trúc L=1 tinh gọn & Benchmark</td>
          <td>3.5 phút</td>
          <td>Slide 6 – 8</td>
          <td>Đột phá cốt lõi: Optuna giảm 78% bộ nhớ đệm, an toàn 1.04 ms.</td>
        </tr>
        <tr>
          <td><strong>Phần 5: Phân tích</strong></td>
          <td>Hiện tượng Seasonal Shift, Parity Test 0.0 lệch & Hạn chế</td>
          <td>2.5 phút</td>
          <td>Slide 9 – 11</td>
          <td>Khách quan, giải thích gap Train-Test bằng dịch chuyển mùa mưa.</td>
        </tr>
        <tr>
          <td><strong>Phần 6: Kết luận</strong></td>
          <td>Tổng kết, Khả năng tái lập 100% & Mở đầu phần Hỏi - Đáp</td>
          <td>1.0 phút</td>
          <td>Slide 12</td>
          <td>Gây ấn tượng kết thúc mạnh mẽ, mời Hội đồng chất vấn.</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<!-- SLIDE 1 & 2 -->
<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 1: TIÊU ĐỀ & GIỚI THIỆU THÀNH VIÊN</span>
    <span class="slide-time">0:00 - 0:45 (45 giây)</span>
  </div>
  <div class="slide-body">
    <div class="slide-bullets">
      <ul>
        <li><strong>Đề tài:</strong> Tối ưu hóa Siêu tham số & Chiều sâu đặc trưng có Ràng buộc Độ trễ cho bài toán Dự báo Lượng mưa Ngắn hạn.</li>
        <li><strong>Bài toán:</strong> Dự báo lượng mưa thời gian thực 1 giờ tới (y_{t+1} mm) tại Trạm Tân Sơn Nhất (WMO 48900).</li>
        <li><strong>Nhóm thực hiện:</strong> Sinh viên nhóm nghiên cứu | Giảng viên hướng dẫn: Thầy/Cô Bộ môn.</li>
      </ul>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Kính thưa Thầy/Cô và các bạn, ngập lụt đô thị do các trận mưa dông cực đoan cục bộ là một trong những thách thức hạ tầng nhức nhối nhất tại TP.HCM. Tuy nhiên, để hệ thống bơm thông minh và các bảng cảnh báo giao thông vận hành hiệu quả, chúng ta không chỉ cần một mô hình dự báo chính xác mà mô hình đó <strong>bắt buộc phải đưa ra kết quả trong vài mili-giây ngay trên thiết bị biên</strong>. Hôm nay, em xin đại diện nhóm trình bày đề tài: <em>'Tối ưu hóa Siêu tham số và Chiều sâu đặc trưng có Ràng buộc Độ trễ cho bài toán Dự báo Lượng mưa Ngắn hạn'</em>."
      </div>
    </div>
  </div>
</div>

<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 2: BỐI CẢNH & THÁCH THỨC ĐỘ TRỄ THỜI GIAN THỰC</span>
    <span class="slide-time">0:45 - 2:00 (1 phút 15 giây)</span>
  </div>
  <div class="slide-body">
    <div class="slide-bullets">
      <ul>
        <li><strong>Ngập úng cục bộ TP.HCM:</strong> Mưa đối lưu nhiệt đới bùng phát rất nhanh; hệ thống điều tiết thoát nước cần dự báo tức thì trước 1 giờ.</li>
        <li><strong>Hạn chế của Cloud ML truyền thống:</strong> Trễ mạng truyền tải hai chiều (round-trip) dao động 50–200 ms, tiềm ẩn nguy cơ mất kết nối khi có bão.</li>
        <li><strong>Ràng buộc Kỹ thuật khắt khe:</strong> Thời gian suy luận đầu-cuối T_e2e ≤ 5.0 ms (với <strong>Biên an toàn nội bộ ≤ 4.5 ms</strong>).</li>
        <li><strong>Đánh đổi kỹ thuật (Engineering Trade-off):</strong> T_e2e = T_feat(L) + T_infer(θ). Nếu tăng số giờ quá khứ L (ví dụ L=72) và tăng độ sâu cây, trễ tiền xử lý và suy luận tăng vọt, lập tức làm vỡ SLA.</li>
      </ul>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Thưa Thầy/Cô, trong các bài toán Machine Learning thông thường, chúng ta thường cố gắng tối đa hóa độ chính xác bằng cách tạo thật nhiều đặc trưng trễ và tăng độ sâu mô hình. Nhưng trên thiết bị biên (Edge IoT), nếu mô hình mất 10 hay 20 mili-giây để tính toán thì hệ thống sẽ tắc nghẽn khi tiếp nhận luồng dữ liệu liên tục từ mạng lưới hàng trăm trạm cảm biến. <strong>Ràng buộc SLA mà nhóm đặt ra là T_e2e ≤ 5.0 ms, đi kèm biên an toàn dự phòng là 4.5 ms</strong>. Điều này tạo ra một bài toán đánh đổi trực tiếp: mô hình phải đủ sâu để nắm bắt quy luật khí tượng, nhưng phải đủ nhẹ để chạy kịp ngân sách thời gian."
      </div>
    </div>
  </div>
</div>

<!-- SLIDE 3 & 4 -->
<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 3: KIỂM TOÁN DỮ LIỆU & DATA QUALITY GATE</span>
    <span class="slide-time">2:00 - 3:30 (1 phút 30 giây)</span>
  </div>
  <div class="slide-body">
    <div class="slide-bullets">
      <ul>
        <li><strong>Nguồn dữ liệu:</strong> Quan trắc hàng giờ Meteostat (Tân Sơn Nhất 48900) gồm: Nhiệt độ, Điểm sương, Độ ẩm, Áp suất, Gió và Mưa.</li>
        <li><strong>Cổng kiểm soát chất lượng (Data Quality Gate ≤ 5% Missing):</strong> Căn cứ trên số liệu kiểm toán khách quan:
          <ul>
            <li><strong>Năm 2021:</strong> Thiếu <strong>100.0%</strong> dữ liệu lượng mưa (trạm chưa gắn sensor đo giờ) → Loại bỏ.</li>
            <li><strong>Năm 2022:</strong> Thiếu <strong>40.23%</strong> (chỉ bắt đầu đo từ giữa năm) → Loại bỏ.</li>
            <li><strong>Chuỗi 3 năm 2023–2025:</strong> 26.304 giờ lịch, tỷ lệ thiếu trung bình chỉ <strong>2.13%</strong> (thỏa mãn chuẩn ≤ 5%) → Tập nghiên cứu chính thức.</li>
          </ul>
        </li>
        <li><strong>Chống rò rỉ (Leakage-free):</strong> Áp dụng Causal Forward-Fill (tối đa 3h); phân chia Train (2023–2024: 17.286 mẫu), Val (H1/2025: 4.057 mẫu), Test (H2/2025: 4.399 mẫu).</li>
      </ul>
    </div>
    <div class="img-box">
      <img src="IMG_MISSING_PLACEHOLDER" alt="Missing rate by year" />
      <div class="caption">Hình 1: Tỷ lệ thiếu hụt dữ liệu theo năm và quyết định kích hoạt Data Quality Gate (≤ 5%)</div>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Để đảm bảo tính trung thực khoa học, nhóm bắt đầu bằng việc kiểm toán toàn diện chất lượng dữ liệu. Kết quả bằng số liệu thực nghiệm cho thấy: <strong>năm 2021 thiếu 100% dữ liệu mưa do trạm chưa gắn sensor, năm 2022 thiếu tới 40.2%</strong>. Nhóm đã kích hoạt Cổng kiểm soát dữ liệu Data Quality Gate, loại bỏ hoàn toàn 2 năm này và chỉ giữ lại chuỗi 3 năm liên tục 2023–2025 với 26.304 giờ, nơi tỷ lệ thiếu hụt chỉ là 2.13%. Toàn bộ tiền xử lý tuân thủ nguyên tắc nhân quả (chỉ nhìn về quá khứ), và tập dữ liệu được chia theo thứ tự thời gian nghiêm ngặt, đảm bảo không có rò rỉ thông tin."
      </div>
    </div>
  </div>
</div>

<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 4: PHÁT BIỂU BÀI TOÁN TỐI ƯU CÓ RÀNG BUỘC</span>
    <span class="slide-time">3:30 - 4:45 (1 phút 15 giây)</span>
  </div>
  <div class="slide-body">
    <div class="slide-bullets">
      <ul>
        <li><strong>Công thức bài toán tối ưu:</strong>
          Tìm nghiệm x = (L, θ) cực tiểu hóa RMSE_val(L, θ) với ràng buộc T_e2e(L, θ) ≤ 5.0 ms (Biên an toàn: ≤ 4.5 ms).
        </li>
        <li><strong>Không gian biến quyết định x = (L, θ) phối hợp:</strong>
          <ul>
            <li><strong>Cửa sổ lịch sử L:</strong> L ∈ {1, 2, 3, 6, 12, 24, 36, 48, 72} giờ (quyết định số đặc trưng: từ 10 biến ở L=1 đến 71 biến ở L=72).</li>
            <li><strong>Siêu tham số LightGBM θ:</strong> n_estimators ∈ [20, 500], max_depth ∈ [2, 12], learning_rate ∈ [0.005, 0.2] (log-uniform), min_child_samples ∈ [5, 100].</li>
          </ul>
        </li>
      </ul>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Bài toán tối ưu của đề tài là <strong>tối ưu hóa đồng thời cả Siêu tham số mô hình lẫn Chiều sâu đặc trưng có ràng buộc</strong>. Biến quyết định trọng yếu ở đây là cửa sổ lịch sử L, biến thiên từ 1 giờ đến 72 giờ. Khi L=1, mô hình chỉ dùng 10 đặc trưng; khi L tăng lên 6 hay 72, số đặc trưng tăng gấp 4 đến 7 lần, kéo theo kích thước bộ nhớ đệm và thời gian trích xuất tăng theo. Mục tiêu là tìm nghiệm x* cực tiểu hóa RMSE trên tập Validation nhưng bắt buộc phải nằm trong vùng khả thi T_e2e ≤ 5.0 ms."
      </div>
    </div>
  </div>
</div>

<!-- SLIDE 5 & 6 -->
<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 5: THIẾT KẾ ĐỐI SÁNH CÔNG BẰNG (FAIR PROTOCOL)</span>
    <span class="slide-time">4:45 - 6:00 (1 phút 15 giây)</span>
  </div>
  <div class="slide-body">
    <div class="slide-bullets">
      <ul>
        <li><strong>Phương pháp chính:</strong> Bayesian Optimization qua <strong>Optuna TPE</strong> (lấy mẫu có điều kiện dựa trên xác suất hậu nghiệm).</li>
        <li><strong>Baseline đối chứng:</strong> <strong>Random Search (RS)</strong> lấy mẫu độc lập chuẩn công nghiệp.</li>
        <li><strong>Nguyên tắc công bằng (Fairness by Design):</strong>
          <ul>
            <li>Cùng không gian tìm kiếm (L, θ) và cùng phân phối tiên nghiệm (log-uniform cho learning_rate).</li>
            <li>Khảo sát trên <strong>5 random seeds độc lập</strong> (42, 123, 456, 789, 999).</li>
            <li>Hai mức ngân sách: <strong>Phase 1</strong> (40 trials/seed = 200 trials) và <strong>Phase 2</strong> (100 trials/seed = 500 trials/phương pháp). Tổng cộng 1.000 trials.</li>
          </ul>
        </li>
      </ul>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Để kết quả so sánh có giá trị học thuật cao, nhóm tuân thủ một giao thức đánh giá cực kỳ công bằng: Cả Optuna TPE và Random Search đều được cấp chung một không gian tìm kiếm, chung cách lấy mẫu logarit cho learning rate, và chạy trên cùng 5 random seeds độc lập. Nhóm khảo sát ở cả hai mức ngân sách: <strong>40 trials/seed (tổng 200 trials)</strong> để xem thuật toán nào bắt nhịp nhanh hơn, và mở rộng lên <strong>100 trials/seed (tổng 500 trials/phương pháp)</strong> để kiểm tra điểm bão hòa hội tụ."
      </div>
    </div>
  </div>
</div>

<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 6: KẾT QUẢ THỰC NGHIỆM: TỶ LỆ KHẢ THI & ĐỘ ỔN ĐỊNH</span>
    <span class="slide-time">6:00 - 7:45 (1 phút 45 giây)</span>
  </div>
  <div class="slide-body">
    <table>
      <thead>
        <tr>
          <th>Tiêu chí so sánh (500 trials / 5 seeds)</th>
          <th>Random Search</th>
          <th>Optuna (TPE)</th>
          <th>Chênh lệch / Đánh giá</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Số nghiệm khả thi SLA (T_e2e ≤ 5.0 ms)</strong></td>
          <td>243 / 500 (48.6%)</td>
          <td class="highlight-optuna">313 / 500 (62.6%)</td>
          <td><strong>Optuna cao hơn +14.0 pp</strong></td>
        </tr>
        <tr>
          <td><strong>Số nghiệm khả thi Biên an toàn (T_e2e ≤ 4.5 ms)</strong></td>
          <td>165 / 500 (33.0%)</td>
          <td class="highlight-optuna">251 / 500 (50.2%)</td>
          <td><strong>Optuna cao hơn +17.2 pp</strong></td>
        </tr>
        <tr>
          <td><strong>Tỷ lệ khả thi Phase 1 (40 trials/seed)</strong></td>
          <td>78 / 200 (39.0%)</td>
          <td class="highlight-optuna">119 / 200 (59.5%)</td>
          <td><strong>Optuna cao hơn +20.5 pp</strong></td>
        </tr>
        <tr>
          <td><strong>Validation RMSE trung bình qua 5 seeds</strong></td>
          <td>0.8306 ± 0.0017</td>
          <td><strong>0.8301 ± 0.0007</strong></td>
          <td>Tương đương (p = 0.6742 > 0.05)</td>
        </tr>
        <tr>
          <td><strong>Độ lệch chuẩn qua các seeds (Std)</strong></td>
          <td>Std = 0.0017</td>
          <td class="highlight-optuna">Std = 0.0007</td>
          <td><strong>Quan sát thấy giảm phương sai 57.6%</strong></td>
        </tr>
      </tbody>
    </table>
    <div class="img-box">
      <img src="IMG_PARETO_PLACEHOLDER" alt="Pareto Frontier" />
      <div class="caption">Hình 2: Biên Pareto Thể hiện Sự đánh đổi giữa Sai số dự báo và Độ trễ (Vùng vi phạm > 5.0 ms bị loại bỏ)</div>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Thưa Thầy/Cô, đây là phát hiện thực nghiệm quan trọng đầu tiên: Ở ngưỡng SLA 5.0 ms, <strong>Optuna tìm ra 313 nghiệm khả thi (62.6%), vượt trội hơn Random Search chỉ đạt 48.6% (+14.0 điểm phần trăm)</strong>. Khi thắt chặt vào biên an toàn 4.5 ms, ưu thế của Optuna càng mở rộng lên +17.2 pp (50.2% so với 33.0%). Về sai số, bề mặt mất mát quanh vùng tối ưu khá phẳng, cả hai đều đạt Val RMSE quanh 0.830 mm/h (p = 0.6742, không có khác biệt có ý nghĩa thống kê về RMSE). Tuy nhiên, độ lệch chuẩn qua 5 seeds của Optuna giảm từ 0.0017 xuống 0.0007, phản ánh phương sai thực nghiệm giảm tới 57.6%."
      </div>
    </div>
  </div>
</div>

<!-- SLIDE 7 & 8 -->
<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 7: PHÁT HIỆN KIẾN TRÚC CỐT LÕI: ĐÁNH ĐỔI KHÔNG GIAN – THỜI GIAN</span>
    <span class="slide-time">7:45 - 9:15 (1 phút 30 giây)</span>
  </div>
  <div class="slide-body">
    <div class="slide-bullets">
      <ul>
        <li><strong>Champion của Optuna (L*=1):</strong> Chỉ dùng 1 giờ quá khứ → <strong>10 đặc trưng</strong> (6 biến trễ t + 4 biến chu kỳ sin/cos của giờ và tháng). Cấu hình: 150 cây, max_depth=12, lr=0.0215, min_child_samples=20.</li>
        <li><strong>Best của Random Search (L*=6):</strong> Đòi hỏi 6 giờ quá khứ → <strong>45 đặc trưng</strong> (thêm trung bình trượt và độ lệch chuẩn trượt). Cấu hình: 200 cây, max_depth=12.</li>
        <li><strong>Ý nghĩa Kỹ thuật Thực tiễn:</strong>
          <ul>
            <li>Cấu hình L=1 giúp <strong>giảm 78% kích thước bộ nhớ đệm trạng thái chuỗi thời gian</strong> so với L=6.</li>
            <li>Triệt tiêu chi phí lưu đệm chuỗi dài trên RAM của vi điều khiển/Edge gateway.</li>
            <li>Bản chất vật lý: Mưa đối lưu nhiệt đới tại TP.HCM mang tính bùng phát cục bộ tức thời, thông tin thời tiết tại thời điểm t kết hợp pha ngày/đêm quan trọng hơn nhiều so với việc duy trì chuỗi lịch sử nhiều ngày.</li>
          </ul>
        </li>
      </ul>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Kính thưa Thầy/Cô, giá trị lớn nhất của Optuna TPE không nằm ở việc giảm sai số thêm 0.001 mm/h, mà nằm ở <strong>phát hiện về tính tinh gọn của kiến trúc</strong>. Trong khi Random Search dừng lại ở cửa sổ L=6 với 45 đặc trưng, Optuna TPE đã tự động phát hiện ra rằng chỉ cần đúng L=1 (10 đặc trưng) là đã đạt được độ chính xác tương đương, thậm chí nhỉnh hơn trên tập Test. <strong>Việc đưa từ 45 đặc trưng về 10 đặc trưng giúp giảm 78% chi phí đệm dữ liệu</strong>, mang giá trị ứng dụng cực kỳ lớn cho việc nhúng mô hình lên các thiết bị giá rẻ tại trạm quan trắc."
      </div>
    </div>
  </div>
</div>

<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 8: BẢNG ĐỐI CHUẨN ĐA MÔ HÌNH & HAI THANG ĐO ĐỘ TRỄ</span>
    <span class="slide-time">9:15 - 10:45 (1 phút 30 giây)</span>
  </div>
  <div class="slide-body">
    <table>
      <thead>
        <tr>
          <th>Mô hình</th>
          <th>Cửa sổ L</th>
          <th>Số feat</th>
          <th>Test RMSE</th>
          <th>Wet RMSE (y>0)</th>
          <th>Rain F1 (≥0.5mm)</th>
          <th>Trễ C-API (P95)</th>
          <th>Skill Score vs Pers</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Persistence (Ngây thơ)</strong></td>
          <td>1</td>
          <td>1</td>
          <td>1.5674</td>
          <td>3.0420</td>
          <td>0.5570</td>
          <td><strong>0.05 ms</strong></td>
          <td>0.00%</td>
        </tr>
        <tr>
          <td><strong>Climatology (Khí hậu)</strong></td>
          <td>1</td>
          <td>2</td>
          <td>1.4285</td>
          <td>2.6167</td>
          <td>0.1088</td>
          <td><strong>0.08 ms</strong></td>
          <td>+8.86%</td>
        </tr>
        <tr>
          <td><strong>Linear Regression</strong></td>
          <td>6</td>
          <td>45</td>
          <td>1.2203</td>
          <td>2.3411</td>
          <td>0.5112</td>
          <td>1.03 ms</td>
          <td>+22.14%</td>
        </tr>
        <tr>
          <td><strong>LightGBM (Mặc định)</strong></td>
          <td>6</td>
          <td>45</td>
          <td>1.2036</td>
          <td>2.3136</td>
          <td>0.5617</td>
          <td>2.28 ms</td>
          <td>+23.21%</td>
        </tr>
        <tr>
          <td><strong>Random Search Best</strong></td>
          <td>6</td>
          <td>45</td>
          <td>1.1891</td>
          <td>2.2937</td>
          <td>0.5778</td>
          <td>1.18 ms</td>
          <td>+24.14%</td>
        </tr>
        <tr>
          <td><strong>Optuna TPE (Champion)</strong></td>
          <td><strong>1</strong></td>
          <td><strong>10</strong></td>
          <td class="highlight-optuna"><strong>1.1774</strong></td>
          <td class="highlight-optuna"><strong>2.2735</strong></td>
          <td class="highlight-optuna"><strong>0.5882</strong></td>
          <td class="highlight-optuna"><strong>1.04 ms</strong></td>
          <td class="highlight-optuna"><strong>+24.88%</strong></td>
        </tr>
      </tbody>
    </table>
    <div class="slide-bullets">
      <ul>
        <li><strong>Làm rõ Hai Thang đo Độ trễ:</strong>
          <ul>
            <li><strong>Khi tìm kiếm:</strong> Đo bằng Scikit-learn wrapper tiêu chuẩn (`predict`) → trung vị P95 ≈ 4.49 ms, đóng vai trò bộ lọc khắc nghiệt ép bộ tối ưu chọn mô hình nhẹ.</li>
            <li><strong>Khi triển khai:</strong> Gọi trực tiếp C-API (`booster_.predict`) → độ trễ giảm xuống <strong>1.04 ms [1.00, 1.15] ms</strong>, tạo biên an toàn vận hành rất rộng (4.5 - 1.04 = 3.46 ms).</li>
          </ul>
        </li>
      </ul>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Trên tập Test độc lập, mô hình Champion của Optuna đạt Test RMSE tốt nhất là 1.1774 mm/h, hệ số kỹ năng (Skill Score) đạt 24.88% so với mô hình ngây thơ. Đặc biệt, chỉ số F1 phát hiện mưa dông đạt 0.5882, cao nhất trong các mô hình.
        Nhóm cũng xin làm rõ sự khác biệt giữa hai thang đo độ trễ: Trong quá trình tìm kiếm, nhóm cố tình dùng wrapper tiêu chuẩn của scikit-learn với trễ trung vị ~4.5 ms để làm 'thước đo nghiêm ngặt', ép thuật toán đào thải cây cồng kềnh. Nhưng khi triển khai thực tế, việc gọi trực tiếp C-API của LightGBM Booster giúp triệt tiêu chi phí bọc Python và <strong>đưa độ trễ thực tế xuống chỉ còn 1.04 mili-giây</strong>, an toàn tuyệt đối dưới mọi ngưỡng SLA."
      </div>
    </div>
  </div>
</div>

<!-- SLIDE 9 & 10 -->
<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 9: GIẢI MÃ HIỆN TƯỢNG DỊCH CHUYỂN MÙA (SEASONAL SHIFT)</span>
    <span class="slide-time">10:45 - 12:00 (1 phút 15 giây)</span>
  </div>
  <div class="slide-body">
    <div class="slide-bullets">
      <ul>
        <li><strong>Đối chiếu sai số 3 tập:</strong>
          <ul>
            <li>Train RMSE = <strong>0.8243 mm/h</strong> | Val RMSE (H1/2025) = <strong>0.8318 mm/h</strong> → Chênh lệch Train–Val chỉ +0.0075 mm/h → <strong>Chứng minh mô hình không hề bị quá khớp (No Overfitting)</strong>.</li>
            <li>Test RMSE (H2/2025) = <strong>1.1774 mm/h</strong> (tăng +0.3531 mm/h).</li>
          </ul>
        </li>
        <li><strong>Bản chất Thống kê Khí tượng:</strong>
          <ul>
            <li>Tập Validation (H1/2025): Mùa khô và giao mùa, độ lệch chuẩn lượng mưa chỉ 1.34 mm/h.</li>
            <li>Tập Test (H2/2025): Cao điểm mùa mưa Tây Nam, độ lệch chuẩn lượng mưa tăng vọt lên <strong>1.91 mm/h (+42.6%)</strong> với nhiều trận mưa dông vũ lượng lớn bất thường (đỉnh điểm tháng 10/2025).</li>
            <li>Khoảng cách sai số trên tập Test phản ánh đúng chuyển dịch phân phối tự nhiên chứ không phải do mô hình suy thoái.</li>
          </ul>
        </li>
      </ul>
    </div>
    <div class="img-box">
      <img src="IMG_ACTUAL_PLACEHOLDER" alt="Actual vs Predicted on Test" />
      <div class="caption">Hình 3: Đối chiếu Lượng mưa Thực tế vs Dự báo trên Tập Test H2/2025 (Bám sát các đỉnh mưa dông cực đoan)</div>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Tại sao sai số trên tập Val chỉ là 0.83 mm/h mà sang tập Test lại tăng lên 1.17 mm/h? Mô hình có bị quá khớp không?
        Câu trả lời là: <strong>Mô hình hoàn toàn không bị quá khớp</strong>, bởi khoảng cách giữa Train và Validation cực kỳ nhỏ, chỉ chênh 0.0075 mm/h. Khoảng cách tăng lên ở tập Test xuất phát từ hiện tượng Chuyển dịch Phân phối Khí hậu theo mùa (Seasonal Climate Shift). Tập Validation nằm ở nửa đầu năm 2025 – mùa khô; trong khi tập Test nằm ở nửa cuối năm 2025 – giai đoạn đỉnh điểm mùa mưa với độ lệch chuẩn lượng mưa tăng tới 42.6%. Khi biên độ dao động của tự nhiên tăng mạnh, sai số tuyệt đối tăng theo là điều hoàn toàn hợp lý về mặt vật lý khí tượng."
      </div>
    </div>
  </div>
</div>

<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 10: ĐẢM BẢO CHẤT LƯỢNG KỸ THUẬT PHẦN MỀM & DEMO</span>
    <span class="slide-time">12:00 - 13:30 (1 phút 30 giây)</span>
  </div>
  <div class="slide-body">
    <div class="slide-bullets">
      <ul>
        <li><strong>Parity Test Tuyệt đối (Ngăn ngừa Train-Serving Skew):</strong>
          <ul>
            <li>Kiểm thử độ tương đương giữa đường ống huấn luyện hàng loạt (`build_features`) và đường ống suy luận đơn mẫu trực tuyến (`extract_single_sample_features`) trên L ∈ {1, 6, 24, 72}.</li>
            <li>Sai lệch vector đặc trưng ≤ 1.9 × 10^-6 và <strong>sai lệch dự báo đầu ra bằng 0.0 tuyệt đối</strong>.</li>
          </ul>
        </li>
        <li><strong>CI/CD Tự động trên GitHub Actions:</strong> Pipeline tự động chạy trên Python 3.13, kiểm thử 6 ca test nghiêm ngặt và demo suy luận. Trạng thái hiện tại: <strong>100% Passed (Green ✔)</strong>.</li>
        <li><strong>Web Dashboard Tương tác Trực quan (`src/server.py`):</strong> Cho phép tùy biến thông số thời tiết, đo độ trễ phần cứng thực tế và đối sánh đồng thời 7 mô hình.</li>
      </ul>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Về mặt kỹ thuật phần mềm, đề tài đảm bảo tính toàn vẹn ở mức sản phẩm thương mại:
        Thứ nhất, nhóm xây dựng Parity Test kiểm tra độ tương đồng giữa lúc huấn luyện offline và lúc suy luận online cho mọi giá trị L từ 1 đến 72. <strong>Kết quả dự báo giữa hai đường ống lệch nhau đúng bằng 0.0</strong>, đảm bảo không có rủi ro 'train-serving skew'.
        Thứ hai, toàn bộ dự án được tích hợp CI/CD tự động trên GitHub Actions với Python 3.13, tự động kiểm thử 6 ca test nghiêm ngặt và đã đạt tích xanh 100%.
        Thứ ba, nhóm đã đóng gói một Web Dashboard trực quan, cho phép người dùng nhập dữ liệu thời tiết và đo độ trễ suy luận tức thời ngay trên trình duyệt."
      </div>
    </div>
  </div>
</div>

<!-- SLIDE 11 & 12 -->
<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 11: GIỚI HẠN NGHIÊN CỨU & ĐỊNH HƯỚNG PHÁT TRIỂN</span>
    <span class="slide-time">13:30 - 14:15 (45 giây)</span>
  </div>
  <div class="slide-body">
    <div class="slide-bullets">
      <ul>
        <li><strong>Độ lệch mẫu do loại bỏ giá trị khuyết theo L (Sample Truncation):</strong> Mỗi L làm mất L dòng đầu khi `dropna` (L=72 mất 71 dòng nhiều hơn L=1). Dù độ lệch chỉ 0.4%, hướng tới sẽ chuẩn hóa bằng cách cắt bỏ cố định L_max = 72 dòng cho mọi cấu hình.</li>
        <li><strong>Kiểm định chéo chuỗi thời gian (Rolling-Origin Cross-Validation):</strong> Đề tài đang dùng phân chia cố định mô phỏng chuyển dịch mùa. Hướng mở rộng là áp dụng Time-Series Rolling-Origin CV qua nhiều chu kỳ năm.</li>
        <li><strong>Mở rộng Mạng lưới Đa trạm (Spatial Modeling):</strong> Mở rộng từ trạm đơn Tân Sơn Nhất sang mạng lưới các trạm đo trên toàn TP.HCM để nắm bắt tính dị thể không gian.</li>
      </ul>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Nhóm cũng nghiêm túc nhìn nhận các giới hạn kỹ thuật để làm định hướng phát triển: Thứ nhất là việc chuẩn hóa tập mẫu cắt bỏ L_max=72 để so sánh các L trên cùng tập dữ liệu bất biến 100%. Thứ hai là mở rộng kiểm định chéo Rolling-Origin theo chuỗi thời gian nhiều năm. Và thứ ba là tích hợp thêm dữ liệu từ các trạm khí tượng lân cận để mô hình hóa tính biến thiên không gian của mưa dông."
      </div>
    </div>
  </div>
</div>

<div class="slide-container">
  <div class="slide-header">
    <span class="slide-num">SLIDE 12: TỔNG KẾT & KẾT LUẬN</span>
    <span class="slide-time">14:15 - 15:00 (45 giây)</span>
  </div>
  <div class="slide-body">
    <div class="key-takeaway-grid">
      <div class="takeaway-card">
        <h4>1. Tối ưu có Ràng buộc</h4>
        <p>Giải quyết trọn vẹn bài toán T_e2e ≤ 5.0 ms. Optuna đạt tỷ lệ khả thi 62.6% (+14 pp so với RS) và 50.2% ở biên an toàn 4.5 ms.</p>
      </div>
      <div class="takeaway-card">
        <h4>2. Đột phá Kiến trúc</h4>
        <p>Phát hiện cửa sổ tối ưu L=1 (chỉ 10 đặc trưng), giảm 78% kích thước bộ nhớ đệm, đạt F1 = 0.5882 và Test RMSE 1.1774 mm/h.</p>
      </div>
      <div class="takeaway-card">
        <h4>3. Độ trễ 1.04 ms</h4>
        <p>Tối ưu C-API booster đưa độ trễ thực tế xuống 1.04 ms, tạo biên an toàn vận hành 3.46 ms chống trễ biến thiên phần cứng.</p>
      </div>
    </div>
    <div class="speaker-box">
      <div class="speaker-header">🎙️ Lời thoại thuyết trình (Speaker Notes)</div>
      <div class="speaker-script">
        "Tóm lại, đề tài đã chứng minh rằng trong bài toán học máy thời gian thực có ràng buộc, việc tối ưu hóa phối hợp giữa chiều sâu đặc trưng L và siêu tham số mô hình thông qua phương pháp Bayes mang lại lợi thế vượt trội về tính tinh gọn và tỷ lệ khả thi của hệ thống. <strong>Em xin chân thành cảm ơn Quý Thầy/Cô đã chú ý lắng nghe và xin kính mời Hội đồng đặt câu hỏi đóng góp cho đề tài!</strong>"
      </div>
    </div>
  </div>
</div>

<!-- SECTION 2: DEFENSE FAQ SHEET -->
<div class="section-divider"></div>
<h2 class="section-header">BỘ CÂU HỎI PHẢN BIỆN "PHÒNG THỦ ĐỒ ÁN" (DEFENSE FAQ SHEET)</h2>
<p style="font-size: 10.5px; color: #64748b; margin-bottom: 12px;">Dự đoán 6 câu hỏi hóc búa nhất từ Hội đồng đánh giá chuyên môn và chiến lược trả lời chuẩn mực:</p>

<div class="faq-container">
  <div class="faq-q">❓ Câu hỏi 1: "Tại sao Val RMSE của Optuna (0.8318) lại nhỉnh hơn một chút so với Random Search (0.8290), vậy sao các bạn lại bảo Optuna tốt hơn?"</div>
  <div class="faq-a">
    <strong>🎯 Chiến lược trả lời:</strong><br>
    "Dạ thưa Thầy/Cô, về mặt sai số, kiểm định Student's t-test qua 5 seeds cho giá trị <strong>p = 0.6742 > 0.05</strong>, nghĩa là về mặt thống kê, hai phương pháp tương đương nhau về RMSE trên Validation, chênh lệch 0.0028 nằm hoàn toàn trong biên độ nhiễu tái huấn luyện do hạt giống ngẫu nhiên.<br>
    Tuy nhiên, Optuna vượt trội hơn ở hai khía cạnh then chốt: (1) <strong>Tính tinh gọn kiến trúc:</strong> Random Search phải dùng L=6 với 45 đặc trưng mới đạt được sai số đó. Còn Optuna phát hiện ra chỉ cần L=1 với đúng 10 đặc trưng, giúp giảm tới 78% kích thước bộ nhớ đệm chuỗi thời gian; (2) <strong>Tỷ lệ khả thi cao hơn rõ rệt:</strong> Optuna đạt 62.6% so với 48.6% của Random Search (+14.0 điểm phần trăm), và nhỉnh hơn trên tập Test độc lập với Skill Score 24.88% so với 24.14%."
  </div>
</div>

<div class="faq-container">
  <div class="faq-q">❓ Câu hỏi 2: "Tại sao lúc tìm kiếm ghi độ trễ hơn 4 ms mà lúc triển khai các bạn lại bảo chỉ có 1.04 ms? Có mâu thuẫn hay không?"</div>
  <div class="faq-a">
    <strong>🎯 Chiến lược trả lời:</strong><br>
    "Dạ thưa Thầy/Cô, hai con số này hoàn toàn nhất quán và phản ánh hai giai đoạn kỹ thuật khác nhau:<br>
    - <strong>Trong lúc tìm kiếm tự động:</strong> Nhóm dùng wrapper tiêu chuẩn `LGBMRegressor.predict` của Scikit-learn (trung vị ~4.5 ms) do overhead kiểm tra định danh cột của Python. Nhóm cố tình giữ thước đo nghiêm ngặt này để làm 'bộ lọc khắc nghiệt' ép thuật toán đào thải các mô hình cồng kềnh.<br>
    - <strong>Khi triển khai thực tế (Production):</strong> Nhóm gọi trực tiếp C-API ngầm định `model.booster_.predict(feat_vector)`, loại bỏ hoàn toàn overhead của Python wrapper và đưa độ trễ xuống <strong>1.04 ms [1.00, 1.15] ms</strong>. Nhờ đó, hệ thống có biên an toàn vận hành rất lớn (gần 3.5 ms) chống hiện tượng giật trễ (jitter) do điều tiết CPU."
  </div>
</div>

<div class="faq-container">
  <div class="faq-q">❓ Câu hỏi 3: "Tại sao các bạn lại bỏ dữ liệu năm 2021 và 2022? Bỏ bớt dữ liệu có làm mô hình kém tin cậy không?"</div>
  <div class="faq-a">
    <strong>🎯 Chiến lược trả lời:</strong><br>
    "Dạ thưa Thầy/Cô, quyết định này được đưa ra dựa trên số liệu kiểm toán khách quan chứ không phải cảm tính: Năm 2021 thiếu <strong>100.0%</strong> dữ liệu mưa theo giờ do trạm chưa gắn sensor đo lượng mưa; năm 2022 thiếu tới <strong>40.23%</strong> do trạm chỉ bắt đầu đo từ giữa năm. Nếu đưa dữ liệu bị khuyết tới 40% đến 100% vào chuỗi thời gian, việc nội suy sẽ tạo ra dữ liệu giả, gây méo mó phân phối nghiêm trọng. Việc áp dụng Data Quality Gate với ngưỡng ≤ 5% để chỉ giữ lại 3 năm hoàn chỉnh 2023–2025 (26.304 giờ, tỷ lệ thiếu chỉ 2.13%) là giải pháp chuẩn mực để đảm bảo chất lượng mô hình."
  </div>
</div>

<div class="faq-container">
  <div class="faq-q">❓ Câu hỏi 4: "Dữ liệu lượng mưa có rất nhiều số 0 (không mưa). Mô hình của các bạn có bị hiện tượng chỉ toàn dự báo 0 để đạt RMSE thấp không?"</div>
  <div class="faq-a">
    <strong>🎯 Chiến lược trả lời:</strong><br>
    "Dạ thưa Thầy/Cô, đây là đặc tính Zero-Inflation của dữ liệu lượng mưa nhiệt đới (>74% số giờ là 0 mm). Nhóm đã lường trước điều này và đánh giá đa chiều: (1) Nhóm đo riêng chỉ số <strong>Wet Test RMSE</strong> trên các giờ thực sự có mưa (y > 0), Champion đạt 2.2735 mm/h, tốt hơn rất nhiều so với Persistence (3.0420 mm/h); (2) Nhóm đo chỉ số <strong>F1-score phát hiện mưa dông</strong> (≥ 0.5 mm/h), Optuna đạt F1 = 0.5882, cao nhất trong tất cả các mô hình, chứng minh mô hình bắt rất chuẩn các đợt mưa thật chứ không hề bị thiên lệch chỉ dự báo không mưa."
  </div>
</div>

<div class="faq-container">
  <div class="faq-q">❓ Câu hỏi 5: "Tại sao không dùng mạng nơ-ron như LSTM hay Transformer cho chuỗi thời gian mà lại chọn LightGBM?"</div>
  <div class="faq-a">
    <strong>🎯 Chiến lược trả lời:</strong><br>
    "Dạ thưa Thầy/Cô, lý do cốt lõi là <strong>ràng buộc độ trễ 5.0 mili-giây trên CPU thiết bị biên</strong>: Mạng LSTM hay Transformer có chi phí tính toán ma trận đơn mẫu trên CPU thường mất từ 15 đến 50 mili-giây, vi phạm nghiêm trọng ngân sách 5.0 ms. Trong khi đó, LightGBM được viết bằng C++, đánh giá cây nhị phân rẽ nhánh tốc độ cao với độ phức tạp O(T × D) chỉ tốn khoảng 0.5 ms suy luận. Hơn nữa, trên dữ liệu dạng bảng khí tượng (tabular data), Gradient Boosted Trees đã được chứng minh qua nhiều nghiên cứu là đạt độ chính xác tương đương hoặc nhỉnh hơn Deep Learning với chi phí tài nguyên thấp hơn hàng chục lần."
  </div>
</div>

<div class="faq-container">
  <div class="faq-q">❓ Câu hỏi 6: "Parity Test giữa Offline batch và Online single-sample có ý nghĩa gì trong thực tế?"</div>
  <div class="faq-a">
    <strong>🎯 Chiến lược trả lời:</strong><br>
    "Dạ thưa Thầy/Cô, trong kỹ thuật triển khai ML (MLOps), một lỗi nghiêm trọng thường gặp là <strong>Train-Serving Skew</strong> (lệch pha giữa huấn luyện và triển khai). Lúc huấn luyện dùng Pandas xử lý hàng loạt trên cả DataFrame lớn; lúc triển khai dữ liệu đến từng giờ một mẫu đơn lẻ và tính toán trên mảng NumPy. Nếu cách tính trung bình trượt hay độ lệch chuẩn lệch nhau dù chỉ một chút, mô hình online sẽ nhận đầu vào sai. Nhóm đã viết unit test kiểm tra parity trên 4 mức L (1, 6, 24, 72) và chứng minh <strong>đầu ra dự báo giữa hai đường ống lệch nhau đúng bằng 0.0</strong>, khẳng định hệ thống khi đưa ra trạm quan trắc thực tế sẽ hoạt động chính xác 100% như nghiên cứu."
  </div>
</div>

</body>
</html>
"""

# Replace base64 image placeholders
html_content = html_content.replace("IMG_MISSING_PLACEHOLDER", b64_missing)
html_content = html_content.replace("IMG_PARETO_PLACEHOLDER", b64_pareto)
html_content = html_content.replace("IMG_ACTUAL_PLACEHOLDER", b64_actual)

print("[*] Writing HTML template...")
HTML_FILE.write_text(html_content, encoding="utf-8")

print(f"[*] Compiling clean Presentation PDF via Microsoft Edge headless to: {OUTPUT_PDF}")
cmd = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={OUTPUT_PDF}",
    f"file:///{HTML_FILE.as_posix()}"
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print(res.stderr)

if OUTPUT_PDF.exists():
    print(f"[+] SUCCESS! Generated Presentation PDF ({OUTPUT_PDF.stat().st_size:,} bytes)")
    if HTML_FILE.exists():
        HTML_FILE.unlink()
else:
    print("[-] FAILED to generate Presentation PDF!")
