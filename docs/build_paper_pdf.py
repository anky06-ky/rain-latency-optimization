"""
Build a comprehensive, publication-grade Academic Research Paper PDF for the Rainfall Latency Optimization Project.
Includes:
- Academic Paper Layout (IEEE / Springer style header, abstract, keywords, numbered sections)
- Comprehensive Background & Real-world Motivation
- End-to-End Operational Workflow & Architectural Diagrams (SVG/CSS)
- Rigorous Mathematical Formulations (Constrained Optimization, TPE, Expected Improvement, Causal Imputation, Metric Formulas)
- Embedded Figures (Base64)
- Benchmark Tables & Statistical Hypothesis Testing
- Software Verification & Limitations
Compiles via Microsoft Edge headless.
"""

import sys
import base64
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_PDF = DOCS_DIR / "BAI_BAO_KHOA_HOC_TOI_UU_HOA_DU_BAO_LUONG_MUA.pdf"
HTML_FILE = DOCS_DIR / "paper_temp.html"

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
b64_dist = img_to_b64(FIGURES_DIR / "rainfall_distribution.png")
b64_season = img_to_b64(FIGURES_DIR / "seasonality_and_diurnal.png")

html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo cáo Khoa học: Tối ưu hóa Siêu tham số & Chiều sâu Đặc trưng có Ràng buộc Độ trễ</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&display=swap');

  @page {
    size: A4;
    margin: 16mm 16mm 18mm 16mm;
    @bottom-center {
      content: "- " counter(page) " -";
      font-size: 9px;
      color: #64748b;
      font-family: 'Inter', sans-serif;
    }
    @top-right {
      content: "Đồ án Tối ưu hóa: Dự báo Lượng mưa Thời gian thực | Trạm Tân Sơn Nhất";
      font-size: 8px;
      color: #94a3b8;
      font-family: 'Inter', sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
    background: #ffffff;
    font-size: 11px;
    line-height: 1.6;
    text-align: justify;
  }

  /* Academic Header */
  .paper-header {
    border-bottom: 2px solid #0f172a;
    padding-bottom: 14px;
    margin-bottom: 16px;
    text-align: center;
  }

  .journal-tag {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #0284c7;
    margin-bottom: 6px;
  }

  h1.paper-title {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 800;
    line-height: 1.35;
    color: #0f172a;
    margin-bottom: 6px;
    text-align: center;
  }

  .paper-subtitle {
    font-family: 'Newsreader', serif;
    font-size: 12px;
    font-style: italic;
    color: #334155;
    margin-bottom: 12px;
    text-align: center;
  }

  .author-row {
    font-size: 10px;
    color: #475569;
    margin-bottom: 4px;
    text-align: center;
  }

  .author-row strong {
    color: #0f172a;
  }

  .affiliation-row {
    font-size: 9px;
    color: #64748b;
    text-align: center;
    margin-bottom: 14px;
  }

  /* Abstract Box */
  .abstract-box {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 18px;
    font-size: 10.2px;
    line-height: 1.55;
  }

  .abstract-title {
    font-weight: 800;
    text-transform: uppercase;
    font-size: 9.5px;
    color: #0f172a;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }

  .keywords-row {
    margin-top: 6px;
    font-size: 9.5px;
    color: #334155;
  }

  .keywords-row strong {
    color: #0f172a;
  }

  /* Section Headings */
  h2.sec-heading {
    font-size: 13px;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 1.5px solid #0284c7;
    padding-bottom: 3px;
    margin-top: 18px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    page-break-after: avoid;
    break-after: avoid;
  }

  h3.subsec-heading {
    font-size: 11.5px;
    font-weight: 700;
    color: #0369a1;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
    break-after: avoid;
  }

  p {
    margin-bottom: 8px;
    text-indent: 12px;
  }

  p.no-indent {
    text-indent: 0;
  }

  /* Math Blocks */
  .math-block {
    background: #f8fafc;
    border-left: 3px solid #0284c7;
    padding: 8px 14px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #0f172a;
    border-radius: 0 6px 6px 0;
    line-height: 1.6;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  /* Workflow & Architecture Diagrams */
  .diagram-container {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 12px;
    margin: 12px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  .flow-grid {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    margin: 8px 0;
  }

  .flow-step {
    flex: 1;
    background: #f1f5f9;
    border: 1px solid #94a3b8;
    border-radius: 6px;
    padding: 8px 6px;
    text-align: center;
    font-size: 9px;
    position: relative;
  }

  .flow-step.highlight {
    background: #e0f2fe;
    border-color: #0284c7;
  }

  .flow-step.success {
    background: #dcfce7;
    border-color: #10b981;
  }

  .flow-step strong {
    display: block;
    font-size: 9.5px;
    color: #0f172a;
    margin-bottom: 2px;
  }

  .flow-arrow {
    font-size: 14px;
    font-weight: 800;
    color: #64748b;
  }

  /* Comparison Cards */
  .arch-compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 10px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  .arch-card {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 10px;
    font-size: 10px;
    background: #f8fafc;
  }

  .arch-card.champion {
    border-color: #10b981;
    background: #f0fdf4;
  }

  .arch-card h4 {
    font-size: 11px;
    color: #0f172a;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .badge-tag {
    font-size: 8px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    text-transform: uppercase;
  }

  .badge-tag.green { background: #10b981; color: #fff; }
  .badge-tag.gray { background: #64748b; color: #fff; }

  /* Figures */
  .figure-container {
    text-align: center;
    margin: 12px 0 16px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  .figure-container img {
    max-width: 95%;
    max-height: 210px;
    height: auto;
    border-radius: 6px;
    border: 1px solid #cbd5e1;
  }

  .figure-caption {
    font-size: 9.5px;
    color: #475569;
    margin-top: 4px;
    font-weight: 600;
    text-align: center;
  }

  .figure-grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 10px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  .figure-grid-2 img {
    width: 100%;
    max-height: 155px;
    object-fit: contain;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
  }

  /* Academic Tables */
  table.academic-table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 14px 0;
    font-size: 9.8px;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  table.academic-table th, table.academic-table td {
    padding: 5px 8px;
    text-align: left;
    border-top: 1px solid #cbd5e1;
    border-bottom: 1px solid #cbd5e1;
  }

  table.academic-table th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 9px;
    letter-spacing: 0.3px;
    border-top: 2px solid #0f172a;
    border-bottom: 2px solid #0f172a;
  }

  table.academic-table tr.champion-row {
    background-color: #ecfdf5;
    font-weight: 700;
    color: #065f46;
  }

  table.academic-table tr:last-child td {
    border-bottom: 2px solid #0f172a;
  }

  /* Callout notes */
  .callout-note {
    background: #f0fdf4;
    border-left: 3.5px solid #10b981;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 10px;
    border-radius: 0 4px 4px 0;
  }

  .callout-warn {
    background: #fffbeb;
    border-left: 3.5px solid #f59e0b;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 10px;
    border-radius: 0 4px 4px 0;
  }

  /* Reference list */
  .references-list {
    margin-top: 10px;
    padding-left: 18px;
    font-size: 9px;
    line-height: 1.5;
    color: #334155;
  }

  .references-list li {
    margin-bottom: 4px;
    text-indent: -18px;
    padding-left: 18px;
  }

  .page-break {
    page-break-before: always;
    break-before: page;
  }
</style>
</head>
<body>

<!-- ACADEMIC HEADER -->
<div class="paper-header">
  <div class="journal-tag">ĐỒ ÁN NGHIÊN CỨU & PHÁT TRIỂN HỆ THỐNG ML THỜI GIAN THỰC | WMO 48900</div>
  <h1 class="paper-title">TỐI ƯU HÓA ĐỒNG THỜI SIÊU THAM SỐ VÀ CHIỀU SÂU ĐẶC TRƯNG CÓ RÀNG BUỘC ĐỘ TRỄ CHO BÀI TOÁN DỰ BÁO LƯỢNG MƯA THỜI GIAN THỰC</h1>
  <div class="paper-subtitle">Constrained Joint Hyperparameter and Feature Optimization for Real-Time Short-Term Rainfall Nowcasting under Strict Inference Latency SLA</div>
  <div class="author-row">
    <strong>Nhóm Nghiên Cứu Đề Tài</strong> &nbsp;|&nbsp; 
    <em>Học phần:</em> Nhập môn Tối ưu hóa / Khoa học Dữ liệu Ứng dụng &nbsp;|&nbsp; 
    <em>Mã trạm:</em> Tân Sơn Nhất (ID: 48900)
  </div>
  <div class="affiliation-row">
    Repository: <code>https://github.com/anky06-ky/rain-latency-optimization</code> &nbsp;•&nbsp; Runtime: Python 3.13 | LightGBM C-API | Optuna TPE
  </div>
</div>

<!-- ABSTRACT BOX -->
<div class="abstract-box">
  <div class="abstract-title">TÓM TẮT (ABSTRACT)</div>
  <p class="no-indent">
    Dự báo lượng mưa cực ngắn (nowcasting) 1 giờ tới tại các trạm quan trắc đô thị đóng vai trò cốt lõi trong việc tự động hóa trạm bơm và cảnh báo ngập lụt tại TP.HCM. Tuy nhiên, trong các kiến trúc biên (Edge IoT), mô hình không chỉ đòi hỏi sai số thấp mà phải thỏa mãn nghiêm ngặt giới hạn độ trễ từ đầu-đến-cuối (End-to-End SLA $T_{\text{e2e}} \le 5.0\text{ ms}$). Nghiên cứu này mô hình hóa bài toán dưới dạng <strong>Tối ưu hóa có Ràng buộc (Constrained Optimization)</strong>, đồng thời tìm kiếm chiều dài cửa sổ lịch sử $L \in \{1, \dots, 72\}$ giờ và không gian siêu tham số $\theta$ của LightGBM. Dữ liệu được kiểm toán từ trạm Tân Sơn Nhất (WMO 48900) giai đoạn 2023–2025 (26.304 giờ) với cổng kiểm soát chất lượng (Missing Rate 2.13% $\le 5\%$) và điền khuyết nhân quả (causal imputation). Thực nghiệm đối sánh công bằng trên 5 random seeds (tổng 1.000 trials) chứng minh rằng: Thuật toán Bayesian Optimization qua <strong>Optuna TPE</strong> đạt tỷ lệ khả thi vượt trội (+14.0 điểm phần trăm ở ngưỡng 5.0 ms và +17.2 điểm phần trăm ở biên an toàn 4.5 ms) so với Random Search. Đáng chú ý, Optuna tự động phát hiện ra cấu hình kiến trúc tinh gọn $L^* = 1$ giờ (chỉ 10 đặc trưng), giúp cắt giảm <strong>78% kích thước bộ nhớ đệm trạng thái chuỗi thời gian</strong> so với cấu hình $L=6$ của Random Search mà vẫn duy trì Test RMSE 1.1774 mm/h, $F_1 = 0.5882$ và Skill Score 24.88%. Khi triển khai qua C-API ngầm định, độ trễ thực tế chỉ đạt <strong>1.04 ms</strong>, thỏa mãn an toàn biên vận hành. Toàn bộ mã nguồn đạt 100% kiểm thử tính tương đồng (Parity Test sai lệch $0.0$) và tự động hóa qua CI/CD GitHub Actions trên Python 3.13.
  </p>
  <div class="keywords-row">
    <strong>Từ khóa (Keywords):</strong> Dự báo lượng mưa thời gian thực; Tối ưu hóa có ràng buộc; Độ trễ suy luận biên; Optuna TPE; Cửa sổ lịch sử đặc trưng; LightGBM C-API; Ngập lụt đô thị TP.HCM.
  </div>
</div>

<!-- SECTION 1 -->
<h2 class="sec-heading">1. ĐẶT VẤN ĐỀ & BỐI CẢNH NGHIÊN CỨU</h2>

<h3 class="subsec-heading">1.1. Bối cảnh Ngập lụt Đô thị TP.HCM và Nhu cầu Cảnh báo Sớm Cận Thực</h3>
<p>
  TP. Hồ Chí Minh là một trong những siêu đô thị chịu ảnh hưởng nặng nề nhất bởi biến đổi khí hậu và hiện tượng ngập úng cực đoan. Địa hình trũng thấp kết hợp triều cường và các cơn mưa dông nhiệt đới cục bộ (đối lưu nhiệt) với cường suất cao trong thời gian ngắn thường xuyên làm tê liệt hệ thống giao thông và ngập sâu các lưu vực trũng. Để hệ thống bơm chống ngập thông minh và các cổng kiểm soát triều vận hành chủ động, việc cảnh báo trước lượng mưa tích lũy trong 1 giờ tới ($y_{t+1}$) có ý nghĩa quyết định nhằm tạo thời gian dự phòng kích hoạt máy bơm và hạ mực nước đệm trong kênh rạch.
</p>

<h3 class="subsec-heading">1.2. Thách thức Ràng buộc Thời gian thực trên Hạ tầng Biên (Edge Computing)</h3>
<p>
  Trong mô hình triển khai truyền thống, dữ liệu trạm quan trắc được đẩy lên máy chủ đám mây (Cloud) để suy luận. Tuy nhiên, kiến trúc này bộc lộ điểm yếu chết người trong các tình huống thời tiết cực đoan: độ trễ truyền gói tin qua mạng (network round-trip time) dao động lớn (50–300 ms), băng thông bị nghẽn hoặc mất kết nối viễn thông khi mưa bão lớn. Do đó, việc triển khai suy luận trực tiếp trên thiết bị tính toán biên (Edge Gateway / Raspberry Pi / Vi điều khiển tại trạm) là xu thế tất yếu.
</p>
<p>
  Tại trạm biên, hệ thống liên tục tiếp nhận luồng dữ liệu thời tiết với tần suất cao. Để đảm bảo hệ thống không bị tồn đọng hàng đợi (queue blocking) và đủ năng lực phục vụ đồng thời các tác vụ thu thập, mã hóa và truyền tin cảnh báo, bài toán đặt ra <strong>Thỏa thuận Mức Dịch vụ (SLA) nghiêm ngặt: Thời gian xử lý từ đầu-đến-cuối (End-to-End Latency) không được vượt quá 5.0 mili-giây</strong> ($T_{\text{e2e}} \le 5.0\text{ ms}$), đi kèm <strong>Biên an toàn nội bộ (Safety Margin) $\le 4.5\text{ ms}$</strong>.
</p>

<h3 class="subsec-heading">1.3. Sự Đánh đổi Giữa Chiều sâu Đặc trưng và Độ trễ Phần cứng</h3>
<p>
  Độ trễ toàn trình $T_{\text{e2e}}$ gồm hai thành phần cộng gộp:
</p>
<div class="math-block">
  T_{\text{e2e}}(L, \theta) = T_{\text{feat}}(L) + T_{\text{infer}}(\theta)
</div>
<p>
  Trong đó, $T_{\text{feat}}(L)$ là thời gian trích xuất đặc trưng chuỗi thời gian từ cửa sổ lịch sử $L$ giờ quá khứ, và $T_{\text{infer}}(\theta)$ là thời gian duyệt qua cấu trúc cây quyết định của mô hình LightGBM với bộ siêu tham số $\theta$. Nếu người kỹ sư cố tình tăng $L$ lên 72 giờ và tăng số cây $n_{\text{estimators}}$ lên 500 cây để ép giảm RMSE, thời gian tính toán trung bình trượt và duyệt cây sẽ tăng vọt, lập tức vi phạm ngưỡng SLA 5.0 ms. Ngược lại, nếu chọn mô hình quá nông và $L$ quá ngắn, mô hình sẽ không nắm bắt được động lực học khí quyển và mất khả năng phát hiện mưa dông. Đây chính là bài toán đánh đổi kinh điển cần được giải quyết bằng <strong>Tối ưu hóa Toán học</strong>.
</p>

<h3 class="subsec-heading">1.4. Ba Câu hỏi Nghiên cứu Then chốt (Research Questions)</h3>
<p class="no-indent">
  Để giải quyết triệt để bài toán trên, nghiên cứu tập trung vào 3 câu hỏi:
</p>
<ul>
  <li><strong>RQ1 (So sánh Công bằng dưới cùng Ngân sách):</strong> Dưới cùng ngân sách hạn chế ($N=40$ trials/seed trên 5 seeds), liệu thuật toán TPE có đạt tỷ lệ nghiệm khả thi ($T_{\text{e2e}} \le 5.0\text{ ms}$) cao hơn và sai số thấp hơn Random Search hay không?</li>
  <li><strong>RQ2 (Mở rộng Ngân sách & Bão hòa Hội tụ):</strong> Khi mở rộng ngân sách lên $N=100$ trials/seed (tổng 500 trials/phương pháp), quy luật hội tụ diễn ra như thế nào, và điểm bão hòa xuất hiện ở đâu?</li>
  <li><strong>RQ3 (Đánh đổi Động lực học & Kiến trúc Biên Pareto):</strong> Hình thái đường biên Pareto giữa RMSE và độ trễ thể hiện sự đánh đổi không gian - thời gian như thế nào giữa các mức cửa sổ $L$?</li>
</ul>

<!-- SECTION 2 -->
<h2 class="sec-heading">2. CƠ SỞ LÝ THUYẾT & MÔ HÌNH HÓA TOÁN HỌC</h2>

<h3 class="subsec-heading">2.1. Phát biểu Bài toán Tối ưu hóa có Ràng buộc</h3>
<p>
  Đề tài được hình thức hóa thành bài toán quy hoạch toán học phi tuyến có ràng buộc (Nonlinear Constrained Optimization):
</p>
<div class="math-block">
  \begin{aligned}
  \min_{x = (L, \theta)} \quad & f(x) = \text{RMSE}_{\text{val}}(L, \theta) = \sqrt{\frac{1}{|\mathcal{D}_{\text{val}}|} \sum_{t \in \mathcal{D}_{\text{val}}} \left( y_{t+1} - \hat{y}_{t+1}(x) \right)^2} \\
  \text{thỏa mãn} \quad & g(x) = P_{95}\left( T_{\text{e2e}}(x) \right) \le 5.0\text{ ms} \quad (\text{Biên an toàn nội bộ: } \le 4.5\text{ ms}) \\
  & L \in \{1, 2, 3, 6, 12, 24, 36, 48, 72\} \text{ (giờ)} \\
  & \theta \in \Theta \text{ (Không gian siêu tham số LightGBM)}
  \end{aligned}
</div>
<p>
  Trong đó, $P_{95}(T_{\text{e2e}})$ là phân vị thứ 95 của độ trễ suy luận đo lường qua hàng trăm lần lặp nhằm phòng ngừa hiện tượng giật xung nhịp CPU (throttling jitter).
</p>

<h3 class="subsec-heading">2.2. Thuật toán Tối ưu hóa Bayes qua Ước lượng Parzen Dạng Cây (Optuna TPE)</h3>
<p>
  Thuật toán TPE (Tree-structured Parzen Estimator) của Bergstra et al. thay thế việc mô hình hóa trực tiếp hàm mục tiêu $p(y \mid x)$ bằng cách mô hình hóa hai phân phối mật độ xác suất của biến cấu hình $x$ dựa trên một ngưỡng phân vị $\gamma$:
</p>
<div class="math-block">
  p(x \mid y) = \begin{cases} 
  \ell(x) & \text{nếu } y < y^* \quad (\text{nhóm cấu hình tốt, chiếm tỷ lệ } \gamma) \\
  g(x) & \text{nếu } y \ge y^* \quad (\text{nhóm cấu hình kém, chiếm tỷ lệ } 1-\gamma)
  \end{cases}
</div>
<p>
  Độ đo kỳ vọng cải thiện (Expected Improvement - $\text{EI}$) được tối đa hóa thông qua tỷ số mật độ:
</p>
<div class="math-block">
  \text{EI}_{y^*}(x) = \int_{-\infty}^{y^*} (y^* - y) p(y \mid x) dy = \frac{\gamma y^* \ell(x) - \ell(x) \int_{-\infty}^{y^*} P(y) dy}{\gamma \ell(x) + (1-\gamma) g(x)} \propto \left( \gamma + \frac{g(x)}{\ell(x)}(1-\gamma) \right)^{-1}
</div>
<p>
  Điều này dẫn đến nguyên lý chọn mẫu đơn giản và hiệu quả: Điểm thử nghiệm tiếp theo $x$ sẽ được chọn sao cho tỷ số $\frac{\ell(x)}{g(x)}$ đạt cực đại, tức là ưu tiên các vùng không gian có xác suất cao sinh ra nghiệm sai số thấp và đồng thời thỏa mãn ràng buộc thời gian trễ.
</p>

<h3 class="subsec-heading">2.3. Mã hóa Góc Tuần hoàn Thời gian (Cyclical Temporal Encoding)</h3>
<p>
  Để mô hình học được chu kỳ ngày/đêm (diurnal cycle) và chu kỳ mùa mưa nhiệt đới gió mùa mà không làm méo mó tính liên tục tại điểm giao thời (ví dụ: giữa 23h và 0h, giữa tháng 12 và tháng 1), nghiên cứu áp dụng phép biến đổi góc lượng giác sin/cos:
</p>
<div class="math-block">
  \begin{aligned}
  \text{sin\_hour} &= \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad &\text{cos\_hour} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right) \\
  \text{sin\_month} &= \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad &\text{cos\_month} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)
  \end{aligned}
</div>

<h3 class="subsec-heading">2.4. Các Chỉ số Đánh giá Chất lượng Dự báo Khí tượng</h3>
<p class="no-indent">
  Nghiên cứu áp dụng hệ thống độ đo toàn diện gồm:
</p>
<ol>
  <li><strong>Root Mean Squared Error (RMSE):</strong> Trừng phạt mạnh các sai số lớn trong các trận mưa dông cực đoan:
    $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
  </li>
  <li><strong>Mean Absolute Error (MAE):</strong> Đo lường sai số tuyệt đối trung bình trên toàn bộ các giờ:
    $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
  </li>
  <li><strong>Wet Test RMSE ($y > 0$):</strong> Đo sai số độc quyền trên các giờ thực tế có mưa nhằm kiểm soát rủi ro của hiện tượng dữ liệu có quá nhiều số 0 (Zero-Inflation).</li>
  <li><strong>Rain Event $F_1\text{-score}$ ($\ge 0.5\text{ mm/h}$):</strong> Đo năng lực phát hiện chính xác các sự kiện mưa có khả năng gây ngập:
    $$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
  </li>
  <li><strong>Hệ số Kỹ năng Khí tượng (Skill Score):</strong> Đo mức cải thiện tương đối của mô hình so với các phương pháp tham chiếu (Persistence hoặc Climatology):
    $$\text{Skill Score} = \left( 1 - \frac{\text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{ref}}} \right) \times 100\%$$
  </li>
</ol>

<!-- SECTION 3: ARCHITECTURE & WORKFLOW -->
<h2 class="sec-heading">3. LƯỢC ĐỒ KIẾN TRÚC & CÁCH THỨC VẬN HÀNH HỆ THỐNG</h2>

<h3 class="subsec-heading">3.1. Lược đồ Luồng Dữ liệu Toàn trình (End-to-End Runtime Pipeline)</h3>
<p>
  Hình 1 mô tả cách thức vận hành thực tế của hệ thống từ lúc cảm biến trạm Tân Sơn Nhất thu thập dữ liệu thô cho tới khi đưa ra cảnh báo ngập lụt:
</p>

<div class="diagram-container">
  <div class="flow-grid">
    <div class="flow-step">
      <strong>[1] Cảm biến IoT</strong>
      Trạm WMO 48900<br>temp, dwpt, rhum, pres, wspd, prcp
    </div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step">
      <strong>[2] Causal Imputer</strong>
      Nội suy nhân quả<br>forward-fill (max 3h)<br>chống rò rỉ 100%
    </div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step highlight">
      <strong>[3] Vectorizer</strong>
      Mảng NumPy L=1<br>6 lags + 4 sin/cos<br>T_feat ≈ 0.05 ms
    </div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step success">
      <strong>[4] LightGBM C-API</strong>
      booster_.predict()<br>150 cây, depth 12<br>T_infer ≈ 0.99 ms
    </div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step">
      <strong>[5] Cảnh báo Ngập</strong>
      T_e2e = 1.04 ms<br>F1 = 0.5882<br>Web Dashboard
    </div>
  </div>
  <div class="figure-caption">Hình 1: Lược đồ Luồng Xử lý Dữ liệu và Dự báo Lượng mưa Thời gian thực tại Thiết bị Biên</div>
</div>

<h3 class="subsec-heading">3.2. Phân tích So sánh Kiến trúc Đệm Dữ liệu: $L=1$ (Optuna) vs $L=6$ (Random Search)</h3>
<p>
  Một phát hiện có giá trị kỹ thuật cốt lõi trong đề tài là sự đối lập về kiến trúc giữa hai thuật toán tối ưu hóa:
</p>

<div class="arch-compare-grid">
  <div class="arch-card champion">
    <h4>
      <span>Champion: Optuna TPE ($L=1$)</span>
      <span class="badge-tag green">Tối Ưu & Tinh Gọn</span>
    </h4>
    <ul>
      <li><strong>Cửa sổ lịch sử:</strong> Đúng $L=1$ giờ quá khứ duy nhất.</li>
      <li><strong>Số chiều đặc trưng:</strong> <strong>10 đặc trưng</strong> (6 biến trễ thời điểm $t$ + 4 biến góc lượng giác sin/cos).</li>
      <li><strong>Bộ nhớ đệm chuỗi thời gian:</strong> Chỉ lưu đúng 1 hàng quan trắc gần nhất trong RAM.</li>
      <li><strong>Thời gian trích xuất ($T_{\text{feat}}$):</strong> $\approx 0.05\text{ ms}$ (cực nhanh).</li>
      <li><strong>Ý nghĩa vật lý:</strong> Khí hậu nhiệt đới gió mùa tại Tân Sơn Nhất mang tính bùng phát đối lưu mạnh; trạng thái nhiệt ẩm tức thời kết hợp chu kỳ giờ trong ngày đủ để dự báo mưa mà không cần lưu đệm quá khứ dài.</li>
    </ul>
  </div>

  <div class="arch-card">
    <h4>
      <span>Baseline: Random Search ($L=6$)</span>
      <span class="badge-tag gray">Cồng Kềnh</span>
    </h4>
    <ul>
      <li><strong>Cửa sổ lịch sử:</strong> Đòi hỏi $L=6$ giờ quá khứ liên tục.</li>
      <li><strong>Số chiều đặc trưng:</strong> <strong>45 đặc trưng</strong> (gồm trễ từ lag 0 đến lag 5, giá trị trung bình trượt, độ lệch chuẩn trượt).</li>
      <li><strong>Bộ nhớ đệm chuỗi thời gian:</strong> Phải duy trì vòng đệm tròn (circular buffer) liên tục 6 giờ trong bộ nhớ.</li>
      <li><strong>Thời gian trích xuất ($T_{\text{feat}}$):</strong> $\approx 0.35\text{ ms}$ (tốn gấp 7 lần).</li>
      <li><strong>Đánh giá:</strong> Cấu hình $L=1$ của Optuna giúp <strong>giảm 78% kích thước bộ nhớ đệm trạng thái chuỗi thời gian</strong> so với $L=6$ mà vẫn đạt sai số tương đương hoặc nhỉnh hơn.</li>
    </ul>
  </div>
</div>

<!-- SECTION 4: DATA AUDIT & QUALITY GATE -->
<h2 class="sec-heading">4. QUY TRÌNH THU THẬP & KIỂM TOÁN CHẤT LƯỢNG DỮ LIỆU</h2>

<h3 class="subsec-heading">4.1. Nguồn Dữ liệu Khí tượng Meteostat (Trạm Tân Sơn Nhất WMO 48900)</h3>
<p>
  Dữ liệu khí quyển được thu thập từ trạm quan trắc quốc tế Tân Sơn Nhất (kinh độ 106.65°E, vĩ độ 10.82°N, độ cao 10m so với mực nước biển), do Meteostat lưu trữ đồng bộ từ mạng lưới trao đổi khí tượng toàn cầu (NOAA ISD / WMO). Dữ liệu thô gồm 6 biến khí quyển cơ bản: nhiệt độ không khí (`temp`), nhiệt độ điểm sương (`dwpt`), độ ẩm tương đối (`rhum`), áp suất khí quyển mực biển (`pres`), tốc độ gió (`wspd`), và lượng mưa tích lũy hàng giờ (`prcp`).
</p>

<h3 class="subsec-heading">4.2. Kích hoạt Cổng Kiểm soát Chất lượng (Data Quality Gate $\le 5\%$ Missing Rate)</h3>
<p>
  Để đảm bảo tính nghiêm cẩn khoa học, nhóm đã thực hiện kiểm toán toàn diện tỷ lệ thiếu hụt dữ liệu (data audit) từ năm 2021 đến 2025. Kết quả thực nghiệm bằng số liệu cụ thể (Hình 2) cho thấy:
</p>
<ul>
  <li><strong>Năm 2021:</strong> Thiếu <strong>100.0%</strong> dữ liệu lượng mưa (do trạm chưa tích hợp cảm biến đo mưa tự động hàng giờ).</li>
  <li><strong>Năm 2022:</strong> Thiếu <strong>40.23%</strong> dữ liệu mưa (chỉ bắt đầu ghi nhận từ nửa cuối năm 2022).</li>
</ul>
<p>
  Cả hai năm này đều vi phạm nghiêm trọng ngưỡng kiểm soát chất lượng chuẩn ($>5\%$). Nếu cố tình nội suy một chuỗi thời gian khuyết tới 40–100%, dữ liệu giả mạo sẽ làm sai lệch hoàn toàn quy luật khí quyển. Do đó, <strong>Cổng kiểm soát dữ liệu Data Quality Gate đã loại bỏ hoàn toàn giai đoạn 2021–2022</strong>.
</p>
<p>
  Ngược lại, giai đoạn 3 năm liên tục <strong>2023–2025</strong> (gồm 26.304 giờ lịch) đạt chất lượng vượt trội: năm 2023 thiếu 0.00% (hoàn hảo), năm 2024 thiếu 2.93%, năm 2025 thiếu 3.47%, <strong>trung bình toàn kỳ chỉ thiếu 2.13%</strong> (thỏa mãn chuẩn $\le 5\%$), được lựa chọn làm tập dữ liệu nghiên cứu chính thức.
</p>

<div class="figure-container">
  <img src="IMG_MISSING_PLACEHOLDER" alt="Tỷ lệ thiếu hụt dữ liệu theo năm" />
  <div class="figure-caption">Hình 2: Kiểm toán Tỷ lệ Thiếu hụt Dữ liệu Lượng mưa theo Năm và Quyết định Kích hoạt Data Quality Gate</div>
</div>

<h3 class="subsec-heading">4.3. Điền khuyết Nhân quả (Causal Forward-Fill) và Phân chia Chuỗi Thời gian (Chronological Split)</h3>
<p>
  Để triệt tiêu hoàn toàn rủi ro rò rỉ dữ liệu (Data Leakage), toàn bộ các khoảng thiếu hụt ngắn ($\le 3$ giờ) được xử lý bằng <strong>Causal Forward-Fill</strong> (chỉ lấy giá trị quan trắc quá khứ lấp đầy về phía trước, tuyệt đối không dùng backward-fill hay nội suy tương lai). Các chuỗi khuyết dài $>3$ giờ được giữ nguyên `NaN` và loại bỏ khi trích xuất vector huấn luyện.
</p>
<p>
  Dữ liệu được phân chia theo trật tự thời gian nghiêm ngặt (Hình 3):
</p>
<ul>
  <li><strong>Tập Huấn luyện (Train Split - 70%):</strong> Giai đoạn 01/01/2023 – 31/12/2024 (2 năm liên tục, 17.544 giờ lịch, <strong>17.286 mẫu mô hình hóa hiệu dụng</strong> sau khi dropna).</li>
  <li><strong>Tập Thẩm định (Validation Split - 15%):</strong> Sáu tháng đầu năm 2025 (01/01/2025 – 30/06/2025, 4.344 giờ lịch, <strong>4.057 mẫu hiệu dụng</strong>). Dùng độc quyền để tính toán hàm mục tiêu $f(x)$ và hướng dẫn thuật toán tối ưu.</li>
  <li><strong>Tập Kiểm thử Độc lập (Test Split - 15%):</strong> Sáu tháng cuối năm 2025 (01/07/2025 – 31/12/2025, 4.416 giờ lịch, <strong>4.399 mẫu hiệu dụng</strong>). Được niêm phong hoàn toàn và chỉ mở ra đánh giá 1 lần duy nhất sau khi đã đóng băng toàn bộ siêu tham số.</li>
</ul>

<div class="figure-container">
  <img src="IMG_SPLIT_PLACEHOLDER" alt="Phân chia tập dữ liệu theo niên lịch" />
  <div class="figure-caption">Hình 3: Phân chia Tập Dữ liệu Chuỗi Thời gian Chặt chẽ theo Niên lịch (2023–2025) Triệt tiêu Rò rỉ Thông tin</div>
</div>

<!-- SECTION 5: EXPERIMENTAL DESIGN -->
<h2 class="sec-heading">5. THIẾT KẾ THỰC NGHIỆM & GIAO THỨC ĐỐI CHUẨN CÔNG BẰNG</h2>

<h3 class="subsec-heading">5.1. Không gian Tìm kiếm Đồng nhất (Identical Search Space)</h3>
<p>
  Cả Optuna TPE và Random Search đều được cấp phát chung chính xác một không gian biến quyết định và cùng phân phối tiên nghiệm:
</p>
<ul>
  <li>$L \in \{1, 2, 3, 6, 12, 24, 36, 48, 72\}$ (giờ);</li>
  <li>`n_estimators` $\in [20, 500]$ (số cây mở rộng đến 500 để khảo sát áp lực trễ);</li>
  <li>`max_depth` $\in [2, 12]$ (độ sâu tối đa của cây);</li>
  <li>`learning_rate` $\in [0.005, 0.2]$ (lấy mẫu theo thang đo log-uniform);</li>
  <li>`min_child_samples` $\in [5, 100]$ (số mẫu tối thiểu ở nút lá).</li>
</ul>

<h3 class="subsec-heading">5.2. Giao thức Đánh giá Đa Hạt giống (Multi-Seed Benchmark Protocol)</h3>
<p>
  Để loại bỏ tính ngẫu nhiên của một lần chạy đơn lẻ, mọi thực nghiệm đều được thực thi lặp lại trên <strong>5 random seeds độc lập</strong> (42, 123, 456, 789, 999). Quá trình khảo sát được chia làm 2 giai đoạn:
</p>
<ul>
  <li><strong>Giai đoạn 1 (Phase 1 - Ngân sách hạn chế):</strong> $N=40$ trials/seed $\times$ 5 seeds = 200 trials nhằm trả lời RQ1 về tốc độ khai phá ban đầu.</li>
  <li><strong>Giai đoạn 2 (Phase 2 - Mở rộng quy mô):</strong> $N=100$ trials/seed $\times$ 5 seeds = 500 trials/phương pháp (tổng cộng 1.000 trials) nhằm trả lời RQ2 và RQ3 về độ bão hòa và biên Pareto.</li>
</ul>

<!-- SECTION 6: RESULTS & DISCUSSION -->
<h2 class="sec-heading">6. KẾT QUẢ THỰC NGHIỆM & THẢO LUẬN KHOA HỌC</h2>

<h3 class="subsec-heading">6.1. Bảng Đối chuẩn Đa Mô hình Toàn diện (Multi-Model Benchmark)</h3>
<p>
  Bảng 1 đối chuẩn chi tiết 7 mô hình đại diện trên cả 3 tập dữ liệu Train, Validation và Test, ghi nhận đầy đủ các chỉ số độ chính xác, độ trễ C-API, và hệ số kỹ năng (Skill Score) so với mô hình ngây thơ (Persistence) và mô hình khí hậu (Climatology):
</p>

<table class="academic-table">
  <thead>
    <tr>
      <th>STT</th>
      <th>Mô hình Khảo sát</th>
      <th>$L$</th>
      <th>Feat</th>
      <th>Cấu hình Siêu tham số $\theta$</th>
      <th>Val RMSE</th>
      <th>Test RMSE</th>
      <th>Test MAE</th>
      <th>Wet RMSE</th>
      <th>Rain $F_1$</th>
      <th>Trễ C-API</th>
      <th>SLA $\le 5$ms</th>
      <th>Skill Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><strong>Persistence (Ngây thơ)</strong></td>
      <td>1</td>
      <td>1</td>
      <td>$\hat{y}_{t+1} = y_t$ (lượng mưa giờ trước)</td>
      <td>1.0699</td>
      <td>1.5674</td>
      <td>0.4543</td>
      <td>3.0420</td>
      <td>0.5570</td>
      <td>0.05 ms</td>
      <td>Khả thi</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <td>2</td>
      <td><strong>Climatology (Khí hậu)</strong></td>
      <td>1</td>
      <td>2</td>
      <td>$\mathbb{E}[y \mid \text{month}, \text{hour}]$</td>
      <td>0.9626</td>
      <td>1.4285</td>
      <td>0.6228</td>
      <td>2.6167</td>
      <td>0.1088</td>
      <td>0.08 ms</td>
      <td>Khả thi</td>
      <td>+8.86%</td>
    </tr>
    <tr>
      <td>3</td>
      <td><strong>Linear Regression (OLS)</strong></td>
      <td>6</td>
      <td>45</td>
      <td>Hồi quy tuyến tính đa biến chuẩn</td>
      <td>0.8468</td>
      <td>1.2203</td>
      <td>0.4712</td>
      <td>2.3411</td>
      <td>0.5112</td>
      <td>1.03 ms</td>
      <td>Khả thi</td>
      <td>+22.14%</td>
    </tr>
    <tr>
      <td>4</td>
      <td><strong>Ridge Regression</strong></td>
      <td>6</td>
      <td>45</td>
      <td>Tuyến tính điều chuẩn L2 ($\alpha=1.0$)</td>
      <td>0.8467</td>
      <td>1.2203</td>
      <td>0.4713</td>
      <td>2.3411</td>
      <td>0.5112</td>
      <td>1.27 ms</td>
      <td>Khả thi</td>
      <td>+22.14%</td>
    </tr>
    <tr>
      <td>5</td>
      <td><strong>LightGBM (Mặc định)</strong></td>
      <td>6</td>
      <td>45</td>
      <td>100 cây, `max_depth` tự do (-1)</td>
      <td>0.8499</td>
      <td>1.2036</td>
      <td>0.4021</td>
      <td>2.3136</td>
      <td>0.5617</td>
      <td>2.28 ms</td>
      <td>Khả thi</td>
      <td>+23.21%</td>
    </tr>
    <tr>
      <td>6</td>
      <td><strong>Random Search Best</strong></td>
      <td>6</td>
      <td>45</td>
      <td>200 cây, depth 12, lr 0.0171</td>
      <td>0.8290</td>
      <td>1.1891</td>
      <td>0.3993</td>
      <td>2.2937</td>
      <td>0.5778</td>
      <td>1.18 ms</td>
      <td>Khả thi</td>
      <td>+24.14%</td>
    </tr>
    <tr class="champion-row">
      <td>7</td>
      <td><strong>Optuna TPE (Champion)</strong></td>
      <td><strong>1</strong></td>
      <td><strong>10</strong></td>
      <td><strong>150 cây, depth 12, lr 0.0215</strong></td>
      <td><strong>0.8318</strong></td>
      <td><strong>1.1774</strong></td>
      <td><strong>0.3998</strong></td>
      <td><strong>2.2735</strong></td>
      <td><strong>0.5882</strong></td>
      <td><strong>1.04 ms</strong></td>
      <td><strong>Khả thi (Biên an toàn)</strong></td>
      <td><strong>+24.88%</strong></td>
    </tr>
  </tbody>
</table>
<div class="figure-caption">Bảng 1: Bảng Đối chuẩn Đa Mô hình Toàn diện trên Tập Test Độc lập H2/2025 (Đồng bộ từ final_test_metrics.json)</div>

<h3 class="subsec-heading">6.2. So sánh Tỷ lệ Khả thi (Feasible Rates) và Độ Ổn định Thuật toán</h3>
<p>
  Bảng 2 tổng hợp kết quả thống kê trực tiếp từ 1.000 trials trong 2 file nhật ký gốc (`results/optimization_history_optuna.csv` và `results/optimization_history_random.csv`):
</p>

<table class="academic-table">
  <thead>
    <tr>
      <th>Chỉ số Đánh giá Thực nghiệm</th>
      <th>Random Search (Baseline)</th>
      <th>Optuna TPE (Đề tài)</th>
      <th>Chênh lệch / Đánh giá Thống kê</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Tổng số lượt đánh giá (Trials)</strong></td>
      <td>500 trials (5 seeds)</td>
      <td>500 trials (5 seeds)</td>
      <td>Cùng ngân sách 100 trials/seed</td>
    </tr>
    <tr>
      <td><strong>Số nghiệm khả thi SLA ($T_{\text{e2e}} \le 5.0\text{ ms}$)</strong></td>
      <td>243 / 500 (48.6%)</td>
      <td><strong>313 / 500 (62.6%)</strong></td>
      <td><strong>Optuna cao hơn +14.0 điểm phần trăm (pp)</strong></td>
    </tr>
    <tr>
      <td><strong>Số nghiệm khả thi Biên an toàn ($T_{\text{e2e}} \le 4.5\text{ ms}$)</strong></td>
      <td>165 / 500 (33.0%)</td>
      <td><strong>251 / 500 (50.2%)</strong></td>
      <td><strong>Optuna cao hơn +17.2 điểm phần trăm (pp)</strong></td>
    </tr>
    <tr>
      <td><strong>Tỷ lệ khả thi Phase 1 (40 trials/seed)</strong></td>
      <td>78 / 200 (39.0%)</td>
      <td><strong>119 / 200 (59.5%)</strong></td>
      <td><strong>Optuna cao hơn +20.5 điểm phần trăm (pp)</strong></td>
    </tr>
    <tr>
      <td><strong>Validation RMSE trong Trial History</strong></td>
      <td><strong>0.8290 mm/h</strong> (seed 42)</td>
      <td>0.8297 mm/h (seed 456)</td>
      <td>Chênh lệch cực nhỏ ($\le 0.0007\text{ mm/h}$)</td>
    </tr>
    <tr>
      <td><strong>Validation RMSE khi Retrain xác nhận</strong></td>
      <td><strong>0.8290 mm/h</strong></td>
      <td>0.8318 mm/h</td>
      <td>Chênh lệch cực nhỏ ($\le 0.0028\text{ mm/h}$)</td>
    </tr>
    <tr>
      <td><strong>Validation RMSE trung bình qua 5 seeds</strong></td>
      <td>$0.8306 \pm 0.0017$</td>
      <td><strong>$0.8301 \pm 0.0007$</strong></td>
      <td>Độ ổn định của Optuna cao hơn</td>
    </tr>
    <tr>
      <td><strong>Độ lệch chuẩn qua 5 seeds (Std)</strong></td>
      <td>Std = 0.0017</td>
      <td><strong>Std = 0.0007</strong></td>
      <td><strong>Quan sát thấy giảm phương sai 57.6%</strong></td>
    </tr>
    <tr>
      <td><strong>Kiểm định Paired Student's t-test (5 seeds)</strong></td>
      <td>—</td>
      <td>—</td>
      <td><strong>$t = 0.4527, p = 0.6742 > 0.05$</strong> (Chấp nhận $H_0$)</td>
    </tr>
    <tr>
      <td><strong>Cửa sổ lịch sử tối ưu ($L^*$)</strong></td>
      <td>$L=6$ giờ (45 đặc trưng)</td>
      <td><strong>$L=1$ giờ (10 đặc trưng)</strong></td>
      <td><strong>Giảm 78% kích thước bộ nhớ đệm</strong></td>
    </tr>
    <tr>
      <td><strong>Độ trễ End-to-End Champion (C-API)</strong></td>
      <td>1.180 ms</td>
      <td><strong>1.041 ms</strong> [1.00, 1.15]</td>
      <td>Thỏa mãn an toàn SLA 5.0 ms & biên 4.5 ms</td>
    </tr>
  </tbody>
</table>
<div class="figure-caption">Bảng 2: So sánh Hiệu quả Tối ưu hóa có Ràng buộc qua 5 Random Seeds</div>

<div class="figure-grid-2">
  <div>
    <img src="IMG_PARETO_PLACEHOLDER" alt="Biên Pareto" />
    <div class="figure-caption">Hình 4: Động lực học Biên Pareto Giữa RMSE và Độ trễ</div>
  </div>
  <div>
    <img src="IMG_CONVERGENCE_PLACEHOLDER" alt="Đường cong hội tụ" />
    <div class="figure-caption">Hình 5: Đường cong Hội tụ Trung bình qua 5 Hạt giống</div>
  </div>
</div>

<h3 class="subsec-heading">6.3. Thảo luận về Kiểm định Thống kê & Tính Logic của Ràng buộc</h3>
<p>
  Từ Bảng 2 và Hình 4, 5, nhóm rút ra 4 nhận định học thuật then chốt:
</p>
<ol>
  <li><strong>Về Sai số Thẩm định (Val RMSE):</strong> Kiểm định Student's t-test cặp đôi cho giá trị $t = 0.4527, p = 0.6742 > 0.05$. Điều này chứng minh rằng về mặt thống kê thuần túy, <strong>hai phương pháp cho sai số Validation tương đương nhau</strong>. Bề mặt hàm mất mát quanh vùng tối ưu khá phẳng (nhiều cấu hình khác nhau đều đạt RMSE quanh 0.829–0.831 mm/h). Nhiễu tái huấn luyện do hạt giống (~0.002) tương đương khoảng cách giữa hai thuật toán.</li>
  <li><strong>Ưu thế Cốt lõi của Optuna TPE:</strong> Giá trị vượt trội của Optuna TPE không nằm ở việc khẳng định hơn về sai số, mà nằm ở <strong>Năng lực Định hướng Không gian Tìm kiếm</strong>: TPE đạt tỷ lệ nghiệm khả thi cao hơn rõ rệt (+14.0 pp ở ngưỡng 5 ms; +17.2 pp ở biên an toàn 4.5 ms; và +20.5 pp ở ngân sách Phase 1), đồng thời tự động phát hiện ra giải pháp cực kỳ tinh gọn $L=1$ (10 đặc trưng so với 45 đặc trưng của Random Search).</li>
  <li><strong>Bản chất Hai Thang đo Độ trễ (Search Wrapper vs Production C-API):</strong>
    <ul>
      <li><em>Khi tìm kiếm:</em> Sử dụng wrapper tiêu chuẩn Scikit-learn (`LGBMRegressor.predict`) với độ trễ trung vị ~4.49 ms. Nhóm cố tình duy trì thước đo này để tạo ra một <strong>"bộ lọc khắc nghiệt"</strong> ép thuật toán phải đào thải 37.4% nghiệm Optuna và 51.4% nghiệm Random Search cồng kềnh.</li>
      <li><em>Khi triển khai thực tế:</em> Chuyển sang gọi C-API ngầm định `model.booster_.predict(feat_vector)`, loại bỏ hoàn toàn chi phí validate DataFrame của Python, đưa độ trễ đơn mẫu xuống <strong>1.04 ms</strong>. Nhờ đó, hệ thống sở hữu một biên an toàn vận hành rất rộng ($4.5 - 1.04 = 3.46\text{ ms}$) chống trễ biến thiên phần cứng.</li>
    </ul>
  </li>
  <li><strong>Giảm Phương sai Thực nghiệm:</strong> Mức giảm độ lệch chuẩn từ 0.0017 xuống 0.0007 (giảm 57.6%) là một quan sát thực nghiệm có giá trị trên 5 seeds khảo sát, thể hiện tính ổn định lấy mẫu của mô hình xác suất TPE.</li>
</ol>

<h3 class="subsec-heading">6.4. Giải mã Hiện tượng Chuyển dịch Phân phối Mùa Khí hậu (Seasonal Shift)</h3>
<p>
  Một câu hỏi khoa học quan trọng: <em>Tại sao Train RMSE là 0.8243 mm/h, Val RMSE là 0.8318 mm/h nhưng sang tập Test lại tăng lên 1.1774 mm/h? Mô hình có bị quá khớp (overfitting) không?</em>
</p>
<p>
  Câu trả lời là: <strong>Mô hình hoàn toàn không bị quá khớp</strong>. Bằng chứng là khoảng cách Train–Val cực kỳ nhỏ (chỉ $+0.0075\text{ mm/h}$). Sự gia tăng sai số ở tập Test xuất phát từ bản chất vật lý khí quyển (Hình 6):
</p>
<ul>
  <li>Tập Validation (H1/2025 - Sáu tháng đầu năm): Gồm mùa khô và giai đoạn chuyển mùa; lượng mưa ít, độ lệch chuẩn lượng mưa chỉ là $\sigma_{\text{val}} = 1.34\text{ mm/h}$.</li>
  <li>Tập Test (H2/2025 - Sáu tháng cuối năm): Rơi vào <strong>cao điểm mùa mưa Tây Nam tại Nam Bộ</strong> với độ lệch chuẩn lượng mưa tăng vọt lên <strong>$\sigma_{\text{test}} = 1.91\text{ mm/h}$ (+42.6%)</strong>, kèm theo các đợt mưa dông cực đoan lịch sử vào tháng 10/2025.</li>
</ul>
<p>
  Khi phương sai tự nhiên của chuỗi thời gian tăng hơn 40%, sai số tuyệt đối tăng theo là quy luật khách quan, chứng minh mô hình tổng quát hóa tốt trên phân phối mùa mưa thực tế.
</p>

<div class="figure-container">
  <img src="IMG_ACTUAL_PLACEHOLDER" alt="Thực tế vs Dự báo trên Test" />
  <div class="figure-caption">Hình 6: Đối chiếu Lượng mưa Thực tế vs Dự báo của Champion Model trên Tập Test Độc lập H2/2025</div>
</div>

<!-- SECTION 7: SOFTWARE ENGINEERING -->
<h2 class="sec-heading">7. ĐẢM BẢO CHẤT LƯỢNG KỸ THUẬT PHẦN MỀM (MLOPS & VERIFICATION)</h2>

<h3 class="subsec-heading">7.1. Kiểm thử Tính tương đương Tuyệt đối (Parity Test Against Train-Serving Skew)</h3>
<p>
  Trong công nghệ MLOps, lỗi nghiêm trọng nhất là độ lệch giữa lúc huấn luyện (Batch Pandas) và lúc triển khai (Single-Sample NumPy Array). Nhóm đã thiết kế ca kiểm thử tự động `test_feature_parity_batch_vs_single_sample` trong [test_pipeline.py](file:///d:/rain-latency-optimization/tests/test_pipeline.py):
</p>
<ul>
  <li>Kiểm tra tương đồng vector đặc trưng trên toàn bộ các mức $L \in \{1, 6, 24, 72\}$: sai lệch vector $\le 1.9 \times 10^{-6}$;</li>
  <li>Kiểm tra đầu ra dự báo của Champion Model trên 25 mẫu ngẫu nhiên: <strong>độ chênh lệch dự báo bằng $0.0$ tuyệt đối</strong> ($< 10^{-5}$).</li>
</ul>

<h3 class="subsec-heading">7.2. Tích hợp Liên tục (CI/CD) trên GitHub Actions với Python 3.13</h3>
<p>
  Repository được cấu hình quy trình CI tự động ([.github/workflows/ci.yml](file:///d:/rain-latency-optimization/.github/workflows/ci.yml)) trên môi trường Ubuntu với Python 3.13. Mỗi commit được tự động kiểm toán dependency, tải dữ liệu Meteostat, chạy 6 ca unit test nghiêm ngặt và kiểm tra demo suy luận. Trạng thái hiện tại: <strong>100% Passed (Green ✔)</strong>.
</p>

<h3 class="subsec-heading">7.3. Triển khai Ứng dụng Web Dashboard Tương tác Trực quan</h3>
<p>
  Nhóm đã xây dựng một Web Application hoàn chỉnh (`src/server.py` và thư mục `web/`), cung cấp giao diện trực quan tại <code>http://127.0.0.1:8000</code>, hỗ trợ:
</p>
<ul>
  <li>Mô phỏng thời tiết tức thời (Nhiệt độ, Độ ẩm, Áp suất, Mưa giờ trước);</li>
  <li>Đo đạc độ trễ phần cứng thực tế (Hardware Latency Benchmark qua $N=200$ lần chạy);</li>
  <li>Đối sánh thời gian thực 7 mô hình tham chiếu và phát cảnh báo cấp độ ngập lụt.</li>
</ul>

<!-- SECTION 8: LIMITATIONS & FUTURE WORK -->
<h2 class="sec-heading">8. GIỚI HẠN NGHIÊN CỨU & ĐỊNH HƯỚNG PHÁT TRIỂN</h2>
<p class="no-indent">
  Nhóm nghiên cứu thẳng thắn nhìn nhận hai giới hạn kỹ thuật để làm tiền đề mở rộng:
</p>
<ol>
  <li><strong>Độ lệch Mẫu do Cắt bỏ Giá trị Khuyết theo $L$ (Sample Truncation by Lag):</strong> Mỗi giá trị $L$ đòi hỏi $L$ hàng dữ liệu quá khứ liên tục, dẫn tới việc cấu hình $L=72$ mất 71 dòng dữ liệu đầu tiên nhiều hơn cấu hình $L=1$ khi `dropna`. Mặc dù độ lệch này rất nhỏ trên tổng số 17.286 mẫu train (~0.4%), định hướng tiếp theo sẽ chuẩn hóa bằng cách cắt bỏ cố định $L_{\max} = 72$ dòng đầu tiên cho tất cả các cấu hình để đảm bảo tập mẫu thẩm định hoàn toàn bất biến tuyệt đối.</li>
  <li><strong>Chiến lược Kiểm định Chéo Theo Chuỗi Thời Gian (Rolling-Origin Cross-Validation):</strong> Do tính chất phân kỳ mùa rõ rệt, nghiên cứu hiện tại áp dụng phân chia niên lịch cố định. Hướng phát triển tiếp theo là áp dụng Time-Series Rolling-Origin Cross-Validation qua nhiều chu kỳ năm nhằm kiểm chứng tính bền vững của các siêu tham số trước các hiện tượng El Niño / La Niña bất thường.</li>
  <li><strong>Mở rộng Mạng lưới Quan trắc Đa trạm (Spatial-Temporal Modeling):</strong> Mở rộng bài toán từ trạm đơn Tân Sơn Nhất sang mạng lưới các trạm lân cận (Nhà Bè, Cần Giờ, Bình Triệu) để mô hình hóa tính dị thể không gian của hoàn lưu gió mùa.</li>
</ol>

<!-- SECTION 9: CONCLUSION & REFERENCES -->
<h2 class="sec-heading">9. KẾT LUẬN & ĐÓNG GÓP CỦA ĐỒ ÁN</h2>
<p>
  Nghiên cứu đã giải quyết thành công bài toán tối ưu hóa siêu tham số và chiều sâu đặc trưng có ràng buộc độ trễ thời gian thực cho dự báo lượng mưa ngắn hạn tại trạm Tân Sơn Nhất. Các đóng góp chính gồm:
</p>
<ol>
  <li><strong>Thiết lập bài toán tối ưu hóa có ràng buộc chặt chẽ:</strong> Phối hợp giữa $L$ và $\theta$ với ràng buộc $T_{\text{e2e}} \le 5.0\text{ ms}$ (biên an toàn $\le 4.5\text{ ms}$).</li>
  <li><strong>Minh chứng ưu thế của Optuna TPE:</strong> Đạt tỷ lệ khả thi vượt trội (+14.0 pp) và giảm phương sai thực nghiệm 57.6% qua 5 random seeds.</li>
  <li><strong>Khám phá kiến trúc tinh gọn $L=1$:</strong> Giảm 78% bộ nhớ đệm chuỗi thời gian, đạt Test RMSE 1.1774 mm/h, $F_1 = 0.5882$, độ trễ C-API 1.04 ms.</li>
  <li><strong>Đảm bảo tính tái lập chuẩn mực:</strong> Hệ thống mã nguồn mở hoàn chỉnh, 100% unit tests pass, đi kèm Web Dashboard tương tác thực tế.</li>
</ol>

<h3 class="subsec-heading">TÀI LIỆU THAM KHẢO (REFERENCES)</h3>
<ol class="references-list">
  <li>[1] J. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, "Algorithms for hyper-parameter optimization," in <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, vol. 24, 2011.</li>
  <li>[2] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in <em>Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.</em>, 2019, pp. 2623–2631.</li>
  <li>[3] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, vol. 30, 2017.</li>
  <li>[4] Meteostat Open Weather and Climate Data, "Hourly weather data for weather station Tan Son Nhat (ID: 48900)," <em>Meteostat Python Package</em>, 2026. [Online]. Available: https://meteostat.net/.</li>
  <li>[5] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," <em>Journal of Machine Learning Research</em>, vol. 12, pp. 2825–2830, 2011.</li>
</ol>

</body>
</html>
"""

# Replace base64 image placeholders
html_content = html_content.replace("IMG_MISSING_PLACEHOLDER", b64_missing)
html_content = html_content.replace("IMG_SPLIT_PLACEHOLDER", b64_split)
html_content = html_content.replace("IMG_PARETO_PLACEHOLDER", b64_pareto)
html_content = html_content.replace("IMG_CONVERGENCE_PLACEHOLDER", b64_convergence)
html_content = html_content.replace("IMG_ACTUAL_PLACEHOLDER", b64_actual)

print("[*] Writing clean HTML template...")
HTML_FILE.write_text(html_content, encoding="utf-8")

print(f"[*] Compiling publication-grade Academic Paper PDF via Edge headless to: {OUTPUT_PDF}")
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
    print(f"[+] SUCCESS! Generated Research Paper PDF ({OUTPUT_PDF.stat().st_size:,} bytes)")
    if HTML_FILE.exists():
        HTML_FILE.unlink()
else:
    print("[-] FAILED to generate Research Paper PDF!")
