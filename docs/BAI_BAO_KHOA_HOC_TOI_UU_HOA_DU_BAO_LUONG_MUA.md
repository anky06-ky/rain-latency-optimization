# TỐI ƯU HÓA ĐỒNG THỜI SIÊU THAM SỐ VÀ CHIỀU SÂU ĐẶC TRƯNG CÓ RÀNG BUỘC ĐỘ TRỄ CHO BÀI TOÁN DỰ BÁO LƯỢNG MƯA THỜI GIAN THỰC TẠI TRẠM TÂN SƠN NHẤT

**English Title:** *Constrained Joint Hyperparameter and Feature Optimization for Real-Time Short-Term Rainfall Nowcasting under Strict Inference Latency SLA*

* **Học phần:** Nhập môn Tối ưu hóa / Khoa học Dữ liệu Ứng dụng
* **Mã trạm quan trắc:** Tân Sơn Nhất (WMO ID: `48900`)
* **Mã nguồn dự án:** [GitHub Repository](https://github.com/anky06-ky/rain-latency-optimization)
* **Môi trường thực nghiệm:** Python 3.13.15 | LightGBM 4.7.0 (C-API) | Scikit-learn 1.9.1 | Optuna 5.0.0

---

## TÓM TẮT (ABSTRACT)

Dự báo lượng mưa cực ngắn (nowcasting) 1 giờ tới tại các trạm quan trắc đô thị đóng vai trò cốt lõi trong việc tự động hóa trạm bơm và cảnh báo ngập lụt tại TP. Hồ Chí Minh. Tuy nhiên, trong các kiến trúc biên (Edge IoT), mô hình không chỉ đòi hỏi sai số thấp mà phải thỏa mãn nghiêm ngặt giới hạn độ trễ từ đầu-đến-cuối (End-to-End SLA $T_{\text{e2e}} \le 5.0\text{ ms}$). Nghiên cứu này mô hình hóa bài toán dưới dạng **Tối ưu hóa có Ràng buộc (Constrained Optimization)**, đồng thời tìm kiếm chiều dài cửa sổ lịch sử $L \in \{1, \dots, 72\}$ giờ và không gian siêu tham số $\theta$ của LightGBM. Dữ liệu được kiểm toán từ trạm Tân Sơn Nhất (WMO 48900) giai đoạn 2023–2025 (26.304 giờ) với cổng kiểm soát chất lượng (Missing Rate 2.13% $\le 5\%$) và điền khuyết nhân quả (causal imputation). 

Thực nghiệm đối sánh công bằng trên 5 random seeds (tổng 1.000 trials) chứng minh rằng: Thuật toán Bayesian Optimization qua **Optuna TPE** đạt tỷ lệ khả thi vượt trội (+14.0 điểm phần trăm ở ngưỡng 5.0 ms và +17.2 điểm phần trăm ở biên an toàn 4.5 ms) so với Random Search. Đáng chú ý, Optuna tự động phát hiện ra cấu hình kiến trúc tinh gọn $L^* = 1$ giờ (10 đặc trưng), giúp cắt giảm **77.8% số chiều đặc trưng** (từ 45 xuống 10) và **83.3% số hàng bộ nhớ đệm chuỗi thời gian** (từ 6 giờ xuống 1 giờ quan trắc) so với cấu hình $L=6$ của Random Search mà vẫn duy trì Test RMSE 1.1774 mm/h, $F_1 = 0.5882$ và Skill Score 24.88%. Khi triển khai qua C-API ngầm định, độ trễ thực tế chỉ đạt **1.04 ms**, thỏa mãn an toàn biên vận hành. Toàn bộ mã nguồn đạt 100% kiểm thử tính tương đồng (Parity Test sai lệch $< 10^{-5}$, tương đương 0.0 trong độ chính xác float32) và tự động hóa qua CI/CD GitHub Actions trên Python 3.13.

**Từ khóa:** *Dự báo lượng mưa thời gian thực; Tối ưu hóa có ràng buộc; Độ trễ suy luận biên; Optuna TPE; Cửa sổ lịch sử đặc trưng; LightGBM C-API; Ngập lụt đô thị TP.HCM.*

---

## 1. ĐẶT VẤN ĐỀ & BỐI CẢNH NGHIÊN CỨU

### 1.1. Bối cảnh Ngập lụt Đô thị TP.HCM và Nhu cầu Cảnh báo Sớm Cận Thực
TP. Hồ Chí Minh là một trong những siêu đô thị chịu ảnh hưởng nặng nề nhất bởi biến đổi khí hậu và hiện tượng ngập úng cực đoan. Địa hình trũng thấp kết hợp triều cường và các cơn mưa dông nhiệt đới cục bộ (đối lưu nhiệt) với cường suất cao trong thời gian ngắn thường xuyên làm tê liệt hệ thống giao thông và ngập sâu các lưu vực trũng. Để hệ thống bơm chống ngập thông minh và các cổng kiểm soát triều vận hành chủ động, việc cảnh báo trước lượng mưa tích lũy trong 1 giờ tới ($y_{t+1}$) có ý nghĩa quyết định nhằm tạo thời gian dự phòng kích hoạt máy bơm và hạ mực nước đệm trong kênh rạch.

### 1.2. Thách thức Ràng buộc Thời gian thực trên Hạ tầng Biên (Edge Computing)
Trong mô hình triển khai truyền thống, dữ liệu trạm quan trắc được đẩy lên máy chủ đám mây (Cloud) để suy luận. Tuy nhiên, kiến trúc này bộc lộ điểm yếu chết người trong các tình huống thời tiết cực đoan: độ trễ truyền gói tin qua mạng (network round-trip time) dao động lớn (50–300 ms), băng thông bị nghẽn hoặc mất kết nối viễn thông khi mưa bão lớn. Do đó, việc triển khai suy luận trực tiếp trên thiết bị tính toán biên (Edge Gateway / Raspberry Pi / Vi điều khiển tại trạm) là xu thế tất yếu.

Tại trạm biên, hệ thống liên tục tiếp nhận luồng dữ liệu thời tiết với tần suất cao. Để đảm bảo hệ thống không bị tồn đọng hàng đợi (queue blocking) và đủ năng lực phục vụ đồng thời các tác vụ thu thập, mã hóa và truyền tin cảnh báo, bài toán đặt ra **Thỏa thuận Mức Dịch vụ (SLA) nghiêm ngặt: Thời gian xử lý từ đầu-đến-cuối (End-to-End Latency) không được vượt quá 5.0 mili-giây** ($T_{\text{e2e}} \le 5.0\text{ ms}$), đi kèm **Biên an toàn nội bộ (Safety Margin) $\le 4.5\text{ ms}$**.

### 1.3. Sự Đánh đổi Giữa Chiều sâu Đặc trưng và Độ trễ Phần cứng
Độ trễ toàn trình $T_{\text{e2e}}$ gồm hai thành phần cộng gộp:

$$T_{\text{e2e}}(L, \theta) = T_{\text{feat}}(L) + T_{\text{infer}}(\theta)$$

Trong đó:
* $T_{\text{feat}}(L)$ là thời gian trích xuất đặc trưng chuỗi thời gian từ cửa sổ lịch sử $L$ giờ quá khứ.
* $T_{\text{infer}}(\theta)$ là thời gian duyệt qua cấu trúc cây quyết định của mô hình LightGBM với bộ siêu tham số $\theta$.

Nếu người kỹ sư cố tình tăng $L$ lên 72 giờ và tăng số cây $n_{\text{estimators}}$ lên 500 cây để ép giảm RMSE, thời gian tính toán trung bình trượt và duyệt cây sẽ tăng vọt, lập tức vi phạm ngưỡng SLA 5.0 ms. Ngược lại, nếu chọn mô hình quá nông và $L$ quá ngắn, mô hình sẽ không nắm bắt được động lực học khí quyển và mất khả năng phát hiện mưa dông. Đây chính là bài toán đánh đổi kinh điển cần được giải quyết bằng **Tối ưu hóa Toán học**.

### 1.4. Ba Câu hỏi Nghiên cứu Then chốt (Research Questions)
* **RQ1 (So sánh Công bằng dưới cùng Ngân sách):** Dưới cùng ngân sách hạn chế ($N=40$ trials/seed trên 5 seeds), liệu thuật toán TPE có đạt tỷ lệ nghiệm khả thi ($T_{\text{e2e}} \le 5.0\text{ ms}$) cao hơn và sai số thấp hơn Random Search hay không?
* **RQ2 (Mở rộng Ngân sách & Bão hòa Hội tụ):** Khi mở rộng ngân sách lên $N=100$ trials/seed (tổng 500 trials/phương pháp), quy luật hội tụ diễn ra như thế nào, và điểm bão hòa xuất hiện ở đâu?
* **RQ3 (Đánh đổi Động lực học & Kiến trúc Biên Pareto):** Hình thái đường biên Pareto giữa RMSE và độ trễ thể hiện sự đánh đổi không gian - thời gian như thế nào giữa các mức cửa sổ $L$?

---

## 2. CƠ SỞ LÝ THUYẾT & MÔ HÌNH HÓA TOÁN HỌC

### 2.1. Phát biểu Bài toán Tối ưu hóa có Ràng buộc
Đề tài được hình thức hóa thành bài toán quy hoạch toán học phi tuyến có ràng buộc (Nonlinear Constrained Optimization):

$$\begin{aligned}
\min_{x = (L, \theta)} \quad & f(x) = \text{RMSE}_{\text{val}}(L, \theta) = \sqrt{\frac{1}{|\mathcal{D}_{\text{val}}|} \sum_{t \in \mathcal{D}_{\text{val}}} \left( y_{t+1} - \hat{y}_{t+1}(x) \right)^2} \\
\text{thỏa mãn} \quad & g(x) = P_{95}\left( T_{\text{e2e}}(x) \right) \le 5.0\text{ ms} \quad (\text{Biên an toàn nội bộ: } \le 4.5\text{ ms}) \\
& L \in \{1, 2, 3, 6, 12, 24, 36, 48, 72\} \text{ (giờ)} \\
& \theta \in \Theta \text{ (Không gian siêu tham số LightGBM)}
\end{aligned}$$

Trong đó, $P_{95}(T_{\text{e2e}})$ là phân vị thứ 95 của độ trễ suy luận đo lường qua hàng trăm lần lặp nhằm phòng ngừa hiện tượng giật xung nhịp CPU (throttling jitter).

### 2.2. Thuật toán Tối ưu hóa Bayes qua Ước lượng Parzen Dạng Cây (Optuna TPE)
Thuật toán TPE (Tree-structured Parzen Estimator) của Bergstra et al. thay thế việc mô hình hóa trực tiếp hàm mục tiêu $p(y \mid x)$ bằng cách mô hình hóa hai phân phối mật độ xác suất của biến cấu hình $x$ dựa trên một ngưỡng phân vị $\gamma$:

$$p(x \mid y) = \begin{cases} 
\ell(x) & \text{nếu } y < y^* \quad (\text{nhóm cấu hình tốt, chiếm tỷ lệ } \gamma) \\
g(x) & \text{nếu } y \ge y^* \quad (\text{nhóm cấu hình kém, chiếm tỷ lệ } 1-\gamma)
\end{cases}$$

Độ đo kỳ vọng cải thiện (Expected Improvement - $\text{EI}$) được tối đa hóa thông qua tỷ số mật độ:

$$\text{EI}_{y^*}(x) = \int_{-\infty}^{y^*} (y^* - y) p(y \mid x) dy = \frac{\gamma y^* \ell(x) - \ell(x) \int_{-\infty}^{y^*} P(y) dy}{\gamma \ell(x) + (1-\gamma) g(x)} \propto \left( \gamma + \frac{g(x)}{\ell(x)}(1-\gamma) \right)^{-1}$$

Điều này dẫn đến nguyên lý chọn mẫu đơn giản và hiệu quả: Điểm thử nghiệm tiếp theo $x$ sẽ được chọn sao cho tỷ số $\frac{\ell(x)}{g(x)}$ đạt cực đại, tức là ưu tiên các vùng không gian có xác suất cao sinh ra nghiệm sai số thấp và đồng thời thỏa mãn ràng buộc thời gian trễ.

### 2.3. Mã hóa Góc Tuần hoàn Thời gian (Cyclical Temporal Encoding)
Để mô hình học được chu kỳ ngày/đêm (diurnal cycle) và chu kỳ mùa mưa nhiệt đới gió mùa mà không làm méo mó tính liên tục tại điểm giao thời (ví dụ: giữa 23h và 0h, giữa tháng 12 và tháng 1), nghiên cứu áp dụng phép biến đổi góc lượng giác sin/cos:

$$\begin{aligned}
\text{sin\_hour} &= \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad &\text{cos\_hour} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right) \\
\text{sin\_month} &= \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad &\text{cos\_month} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)
\end{aligned}$$

### 2.4. Các Chỉ số Đánh giá Chất lượng Dự báo Khí tượng
Nghiên cứu áp dụng hệ thống độ đo toàn diện gồm:
1. **Root Mean Squared Error (RMSE):** Trừng phạt mạnh các sai số lớn trong các trận mưa dông cực đoan:
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
2. **Mean Absolute Error (MAE):** Đo lường sai số tuyệt đối trung bình trên toàn bộ các giờ:
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
3. **Wet Test RMSE ($y > 0$):** Đo sai số độc quyền trên các giờ thực tế có mưa nhằm kiểm soát rủi ro của hiện tượng dữ liệu có quá nhiều số 0 (Zero-Inflation).
4. **Rain Event $F_1\text{-score}$ ($\ge 0.5\text{ mm/h}$):** Đo năng lực phát hiện chính xác các sự kiện mưa có khả năng gây ngập:
   $$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **Hệ số Kỹ năng Khí tượng (Skill Score):** Đo mức cải thiện tương đối của mô hình so với các phương pháp tham chiếu (Persistence hoặc Climatology):
   $$\text{Skill Score} = \left( 1 - \frac{\text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{ref}}} \right) \times 100\%$$

---

## 3. LƯỢC ĐỒ KIẾN TRÚC & CÁCH THỨC VẬN HÀNH HỆ THỐNG

### 3.1. Lược đồ Luồng Dữ liệu Toàn trình (End-to-End Runtime Pipeline)

```
+----------------------------------------------------------------------------------------------------+
|                                   LUỒNG SUY LUẬN THỜI GIAN THỰC (EDGE)                             |
+----------------------------------------------------------------------------------------------------+
|  [Trạm Khí tượng WMO 48900]                                                                       |
|  Cảm biến đo hàng giờ: temp, dwpt, rhum, pres, wspd, prcp (tại thời điểm t)                       |
|                                       │                                                            |
|                                       ▼                                                            |
|  [Bộ Điền Khuyết Nhân Quả (Causal Imputer)]                                                       |
|  Causal Forward-Fill (tối đa 3h) - Tuyệt đối không dùng thông tin tương lai                       |
|                                       │                                                            |
|                                       ▼                                                            |
|  [Bộ Trích Xuất Vector Đặc Trưng Đơn Mẫu (Single-Sample Vectorizer)]                              |
|  Cửa sổ tối ưu L = 1h: 6 biến thời tiết trễ t + 4 biến mã hóa chu kỳ (sin/cos hour & month)      |
|  Thời gian thực thi: T_feat ≈ 0.05 ms                                                             |
|                                       │                                                            |
|                                       ▼                                                            |
|  [Mô hình Champion LightGBM C-API]                                                                |
|  model.booster_.predict(feat_vector) (150 cây, max_depth=12, n_jobs=1)                            |
|  Thời gian suy luận: T_infer ≈ 0.99 ms                                                            |
|                                       │                                                            |
|                                       ▼                                                            |
|  [Bộ Phân Loại Cảnh Báo & Web Dashboard]                                                          |
|  Độ trễ toàn trình: T_e2e = 1.04 ms  [1.00, 1.15] ms (SLA <= 5.0 ms, Biên an toàn <= 4.5 ms)    |
|  Dự báo lượng mưa y_{t+1} (mm/h) | Phân cấp ngập lụt | Hiển thị HTTP Dashboard                    |
+----------------------------------------------------------------------------------------------------+
```

### 3.2. Phân tích So sánh Kiến trúc Đệm Dữ liệu: $L=1$ (Optuna) vs $L=6$ (Random Search)

| Tiêu chí Kiến trúc | Cấu hình Random Search ($L=6$) | Cấu hình Optuna Champion ($L=1$) | Đánh giá Kỹ thuật |
| :--- | :---: | :---: | :--- |
| **Chiều dài Cửa sổ Lịch sử ($L$)** | 6 giờ quá khứ liên tục | **1 giờ quá khứ duy nhất** | Optuna tối giản thời gian đệm |
| **Số chiều Vector Đặc trưng** | 45 đặc trưng (lags + rolling) | **10 đặc trưng** (lags + sin/cos) | **Giảm 77.8% số chiều đặc trưng** (10 vs 45) |
| **Bộ nhớ đệm Chuỗi Thời gian (RAM)** | Circular Buffer chứa 6 bản ghi | Lưu đúng 1 bản ghi gần nhất | **Giảm 83.3% số hàng đệm** (1 vs 6 giờ quan trắc) |
| **Thời gian Trích xuất ($T_{\text{feat}}$)** | $\approx 0.35\text{ ms}$ | $\approx \mathbf{0.05\text{ ms}}$ | Tốc độ trích xuất nhanh gấp 7 lần |
| **Ý nghĩa Vật lý Khí quyển** | Thừa thãi thông tin quá khứ | Nắm bắt đúng trạng thái nhiệt ẩm | Mưa đối lưu nhiệt đới bùng phát nhanh |

---

## 4. QUY TRÌNH THU THẬP & KIỂM TOÁN CHẤT LƯỢNG DỮ LIỆU

### 4.1. Nguồn Dữ liệu Khí tượng Meteostat (Trạm Tân Sơn Nhất WMO 48900)
Dữ liệu khí quyển được thu thập từ trạm quan trắc quốc tế Tân Sơn Nhất (kinh độ 106.65°E, vĩ độ 10.82°N, độ cao 10m so với mực nước biển), do Meteostat lưu trữ đồng bộ từ mạng lưới trao đổi khí tượng toàn cầu (NOAA ISD / WMO). Dữ liệu thô gồm 6 biến khí quyển cơ bản: nhiệt độ không khí (`temp`), nhiệt độ điểm sương (`dwpt`), độ ẩm tương đối (`rhum`), áp suất khí quyển mực biển (`pres`), tốc độ gió (`wspd`), và lượng mưa tích lũy hàng giờ (`prcp`).

### 4.2. Kích hoạt Cổng Kiểm soát Chất lượng (Data Quality Gate $\le 5\%$ Missing Rate)
Để đảm bảo tính nghiêm cẩn khoa học, nhóm đã thực hiện kiểm toán toàn diện tỷ lệ thiếu hụt dữ liệu (data audit) từ năm 2021 đến 2025:
* **Năm 2021:** Thiếu **100.0%** dữ liệu lượng mưa (do trạm chưa tích hợp cảm biến đo mưa tự động hàng giờ).
* **Năm 2022:** Thiếu **40.23%** dữ liệu mưa (chỉ bắt đầu ghi nhận từ nửa cuối năm 2022).

Cả hai năm này đều vi phạm nghiêm trọng ngưỡng kiểm soát chất lượng chuẩn ($>5\%$). Nếu cố tình nội suy một chuỗi thời gian khuyết tới 40–100%, dữ liệu giả mạo sẽ làm sai lệch hoàn toàn quy luật khí quyển. Do đó, **Cổng kiểm soát dữ liệu Data Quality Gate đã loại bỏ hoàn toàn giai đoạn 2021–2022**.

Ngược lại, giai đoạn 3 năm liên tục **2023–2025** (gồm 26.304 giờ lịch) đạt chất lượng vượt trội: năm 2023 thiếu 0.00% (hoàn hảo), năm 2024 thiếu 2.93%, năm 2025 thiếu 3.47%, **trung bình toàn kỳ chỉ thiếu 2.13%** (thỏa mãn chuẩn $\le 5\%$), được lựa chọn làm tập dữ liệu nghiên cứu chính thức.

![Kiểm toán tỷ lệ thiếu hụt](results/figures/missing_rate_by_year.png)

### 4.3. Điền khuyết Nhân quả (Causal Forward-Fill) và Phân chia Chuỗi Thời gian (Chronological Split)
Để triệt tiêu hoàn toàn rủi ro rò rỉ dữ liệu (Data Leakage), toàn bộ các khoảng thiếu hụt ngắn ($\le 3$ giờ) được xử lý bằng **Causal Forward-Fill** (chỉ lấy giá trị quan trắc quá khứ lấp đầy về phía trước, tuyệt đối không dùng backward-fill hay nội suy tương lai). Các chuỗi khuyết dài $>3$ giờ được giữ nguyên `NaN` và loại bỏ khi trích xuất vector huấn luyện.

Dữ liệu được phân chia theo trật tự thời gian nghiêm ngặt:
* **Tập Huấn luyện (Train Split - 70%):** Giai đoạn 01/01/2023 – 31/12/2024 (2 năm liên tục, 17.544 giờ lịch, **17.286 mẫu mô hình hóa hiệu dụng** sau khi dropna).
* **Tập Thẩm định (Validation Split - 15%):** Sáu tháng đầu năm 2025 (01/01/2025 – 30/06/2025, 4.344 giờ lịch, **4.057 mẫu hiệu dụng**). Dùng độc quyền để tính toán hàm mục tiêu $f(x)$ và hướng dẫn thuật toán tối ưu.
* **Tập Kiểm thử Độc lập (Test Split - 15%):** Sáu tháng cuối năm 2025 (01/07/2025 – 31/12/2025, 4.416 giờ lịch, **4.399 mẫu hiệu dụng**). Được niêm phong hoàn toàn và chỉ mở ra đánh giá 1 lần duy nhất sau khi đã đóng băng toàn bộ siêu tham số.

![Phân chia tập dữ liệu theo niên lịch](results/figures/train_val_test_split.png)

---

## 5. THIẾT KẾ THỰC NGHIỆM & GIAO THỨC ĐỐI CHUẨN CÔNG BẰNG

### 5.1. Không gian Tìm kiếm Đồng nhất (Identical Search Space)
Cả Optuna TPE và Random Search đều được cấp phát chung chính xác một không gian biến quyết định và cùng phân phối tiên nghiệm:
* $L \in \{1, 2, 3, 6, 12, 24, 36, 48, 72\}$ (giờ);
* `n_estimators` $\in [20, 500]$ (số cây mở rộng đến 500 để khảo sát áp lực trễ);
* `max_depth` $\in [2, 12]$ (độ sâu tối đa của cây);
* `learning_rate` $\in [0.005, 0.2]$ (lấy mẫu theo thang đo log-uniform);
* `min_child_samples` $\in [5, 100]$ (số mẫu tối thiểu ở nút lá).

### 5.2. Giao thức Đánh giá Đa Hạt giống (Multi-Seed Benchmark Protocol)
Để loại bỏ tính ngẫu nhiên của một lần chạy đơn lẻ, mọi thực nghiệm đều được thực thi lặp lại trên **5 random seeds độc lập** (42, 123, 456, 789, 999). Quá trình khảo sát được chia làm 2 giai đoạn:
* **Giai đoạn 1 (Phase 1 - Ngân sách hạn chế):** $N=40$ trials/seed $\times$ 5 seeds = 200 trials nhằm trả lời RQ1 về tốc độ khai phá ban đầu.
* **Giai đoạn 2 (Phase 2 - Mở rộng quy mô):** $N=100$ trials/seed $\times$ 5 seeds = 500 trials/phương pháp (tổng cộng 1.000 trials) nhằm trả lời RQ2 và RQ3 về độ bão hòa và biên Pareto.

---

## 6. KẾT QUẢ THỰC NGHIỆM & THẢO LUẬN KHOA HỌC

### 6.1. Bảng Đối chuẩn Đa Mô hình Toàn diện (Multi-Model Benchmark)

| STT | Mô hình Khảo sát | $L$ | Feat | Cấu hình Siêu tham số $\theta$ | Val RMSE | Test RMSE | Test MAE | Wet RMSE | Rain $F_1$ | Trễ C-API | SLA $\le 5$ms | Skill Score |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Persistence (Ngây thơ)** | 1 | 1 | $\hat{y}_{t+1} = y_t$ (mưa giờ trước) | 1.0699 | 1.5674 | 0.4543 | 3.0420 | 0.5570 | **0.05 ms** | Khả thi | 0.00% |
| 2 | **Climatology (Khí hậu)** | 1 | 2 | $\mathbb{E}[y \mid \text{month}, \text{hour}]$ | 0.9626 | 1.4285 | 0.6228 | 2.6167 | 0.1088 | **0.08 ms** | Khả thi | +8.86% |
| 3 | **Linear Regression (OLS)** | 6 | 45 | Hồi quy tuyến tính chuẩn | 0.8468 | 1.2203 | 0.4712 | 2.3411 | 0.5112 | 1.03 ms | Khả thi | +22.14% |
| 4 | **Ridge Regression** | 6 | 45 | Tuyến tính L2 ($\alpha=1.0$) | 0.8467 | 1.2203 | 0.4713 | 2.3411 | 0.5112 | 1.27 ms | Khả thi | +22.14% |
| 5 | **LightGBM (Mặc định)** | 6 | 45 | 100 cây, `max_depth` tự do | 0.8499 | 1.2036 | 0.4021 | 2.3136 | 0.5617 | 2.28 ms | Khả thi | +23.21% |
| 6 | **Random Search Best** | 6 | 45 | 200 cây, depth 12, lr 0.0171 | **0.8290** | 1.1891 | 0.3993 | 2.2937 | 0.5778 | 1.18 ms | Khả thi | +24.14% |
| 7 | **Optuna TPE (Champion)** | **1** | **10** | **150 cây, depth 12, lr 0.0215** | **0.8318** | **1.1774** | **0.3998** | **2.2735** | **0.5882** | **1.04 ms** | **Khả thi (Biên an toàn)** | **+24.88%** |

### 6.2. So sánh Tỷ lệ Khả thi (Feasible Rates) và Độ Ổn định Thuật toán

| Chỉ số Đánh giá Thực nghiệm | Random Search (Baseline) | Optuna TPE (Đề tài) | Chênh lệch / Đánh giá Thống kê |
| :--- | :---: | :---: | :--- |
| **Tổng số lượt đánh giá (Trials)** | 500 trials (5 seeds) | 500 trials (5 seeds) | Cùng ngân sách 100 trials/seed |
| **Số nghiệm khả thi SLA ($T_{\text{e2e}} \le 5.0\text{ ms}$)** | 243 / 500 (48.6%) | **313 / 500 (62.6%)** | **Optuna cao hơn +14.0 điểm phần trăm (pp)** |
| **Số nghiệm khả thi Biên an toàn ($T_{\text{e2e}} \le 4.5\text{ ms}$)** | 165 / 500 (33.0%) | **251 / 500 (50.2%)** | **Optuna cao hơn +17.2 điểm phần trăm (pp)** |
| **Tỷ lệ khả thi Phase 1 (40 trials/seed)** | 78 / 200 (39.0%) | **119 / 200 (59.5%)** | **Optuna cao hơn +20.5 điểm phần trăm (pp)** |
| **Kiểm định Fisher's Exact Test (Feasible Rate SLA)** | — | — | **$p = 1.09 \times 10^{-5} < 0.001$, OR = 1.77** (Bác bỏ $H_0$, 95% CI: $[+7.9\%, +20.1\%]$) |
| **Validation RMSE trong Trial History** | **0.8290 mm/h** (seed 42) | 0.8297 mm/h (seed 456) | Chênh lệch cực nhỏ ($\le 0.0007\text{ mm/h}$) |
| **Validation RMSE khi Retrain xác nhận** | **0.8290 mm/h** | 0.8318 mm/h | Chênh lệch cực nhỏ ($\le 0.0028\text{ mm/h}$) |
| **Validation RMSE trung bình qua 5 seeds** | $0.8306 \pm 0.0017$ | **$0.8301 \pm 0.0007$** | Độ ổn định của Optuna cao hơn |
| **Độ lệch chuẩn qua 5 seeds (Std)** | Std = 0.0017 | **Std = 0.0007** | **Quan sát thấy giảm phương sai 57.6%** |
| **Kiểm định Paired Student's t-test (5 seeds)** | — | — | **$t = 0.4527, p = 0.6742 > 0.05$** (Không bác bỏ $H_0$) |
| **Cửa sổ lịch sử tối ưu ($L^*$)** | $L=6$ giờ (45 đặc trưng) | **$L=1$ giờ (10 đặc trưng)** | **Giảm 77.8% số chiều đặc trưng, giảm 83.3% số hàng đệm** |
| **Độ trễ End-to-End Champion (C-API)** | 1.180 ms | **1.041 ms** [1.00, 1.15] | Thỏa mãn an toàn SLA 5.0 ms & biên 4.5 ms |

![Biên Pareto](results/figures/pareto_frontier.png)
![Đường cong hội tụ](results/figures/convergence_comparison.png)

### 6.3. Thảo luận về Kiểm định Thống kê & Tính Logic của Ràng buộc
1. **Về Sai số Thẩm định (Val RMSE):** Kiểm định Student's t-test cặp đôi cho giá trị $t = 0.4527, p = 0.6742 > 0.05$. Điều này chứng minh rằng về mặt thống kê, **không có đủ bằng chứng để bác bỏ $H_0$ (hai phương pháp cho sai số Validation tương đương nhau)**. Bề mặt hàm mất mát quanh vùng tối ưu khá phẳng (nhiều cấu hình khác nhau đều đạt RMSE quanh 0.829–0.831 mm/h). Nhiễu tái huấn luyện do hạt giống (~0.002) tương đương khoảng cách giữa hai thuật toán.
2. **Ưu thế Cốt lõi & Kiểm định Ý nghĩa Thống kê của Tỷ lệ Khả thi:** Giá trị vượt trội của Optuna TPE nằm ở **Năng lực Định hướng Không gian Tìm kiếm**: TPE đạt tỷ lệ nghiệm khả thi cao hơn rõ rệt (+14.0 pp ở ngưỡng 5 ms; +17.2 pp ở biên an toàn 4.5 ms; và +20.5 pp ở ngân sách Phase 1). Kiểm định Fisher's exact test trên toàn bộ 1.000 trials (500 Optuna vs 500 RS) xác nhận mức tăng +14.0 pp đạt mức ý nghĩa thống kê cực kỳ thuyết phục ($p = 1.09 \times 10^{-5} \ll 0.001$, Pearson $\chi^2 = 19.286$, Odds Ratio = 1.77, khoảng tin cậy 95% cho mức chênh lệch là $[+7.9\%, +20.1\%]$). Khi phân rã theo 5 hạt giống độc lập, Optuna vượt trội ở 4/5 seeds (với mức chênh lệch lên tới +40 pp ở seed 42; kiểm định Wilcoxon signed-rank $W = 2.0, p = 0.1875$, paired $t$-test $p = 0.1553$). Đồng thời, thuật toán tự động phát hiện ra giải pháp cực kỳ tinh gọn $L=1$ (10 đặc trưng so với 45 đặc trưng của Random Search).
3. **Bản chất Hai Thang đo Độ trễ (Search Wrapper vs Production C-API):**
   * *Khi tìm kiếm (Search Phase):* Ràng buộc cho mỗi trial $x$ là $g(x) = P_{95}(T_{\text{e2e}}(x)) \le 5.0\text{ ms}$, đo lường phân vị thứ 95 qua 100 lần lặp. Thước đo được thực thi qua Scikit-learn wrapper tiêu chuẩn (`LGBMRegressor.predict`) để phản ánh quy trình AutoML thông thường. Do chi phí kiểm tra kiểu và DataFrame wrapper của Python, **trung vị của các giá trị phân vị $P_{95}$ trên toàn bộ 500 trials trong nhật ký tìm kiếm là ~4.49 ms** (nhiều trial vượt từ 5 ms đến 40 ms khi $L$ lớn và cây sâu). Ràng buộc $P_{95} \le 5.0\text{ ms}$ đóng vai trò một **"bộ lọc khắc nghiệt"** ép thuật toán phải đào thải 37.4% nghiệm Optuna và 51.4% nghiệm Random Search cồng kềnh.
   * *Khi triển khai thực tế (Production Phase):* Chuyển sang gọi C-API ngầm định `model.booster_.predict(feat_vector)`, loại bỏ hoàn toàn chi phí validate DataFrame của Python, đưa độ trễ đơn mẫu xuống **1.04 ms** (range [1.00, 1.15] ms). Nhờ đó, hệ thống sở hữu một biên an toàn vận hành rất rộng ($4.5 - 1.04 = 3.46\text{ ms}$) chống trễ biến thiên phần cứng.
4. **Giảm Phương sai Thực nghiệm:** Mức giảm độ lệch chuẩn từ 0.0017 xuống 0.0007 (giảm 57.6%) là một quan sát thực nghiệm có giá trị trên 5 seeds khảo sát, thể hiện tính ổn định lấy mẫu của mô hình xác suất TPE.

### 6.4. Giải mã Hiện tượng Chuyển dịch Phân phối Mùa Khí hậu (Seasonal Shift)
Một câu hỏi khoa học quan trọng: *Tại sao Train RMSE là 0.8243 mm/h, Val RMSE là 0.8318 mm/h nhưng sang tập Test lại tăng lên 1.1774 mm/h? Mô hình có bị quá khớp (overfitting) không?*

Câu trả lời là: **Mô hình hoàn toàn không bị quá khớp**. Bằng chứng là khoảng cách Train–Val cực kỳ nhỏ (chỉ $+0.0075\text{ mm/h}$). Sự gia tăng sai số ở tập Test xuất phát từ bản chất vật lý khí quyển:
* Tập Validation (H1/2025 - Sáu tháng đầu năm): Gồm mùa khô và giai đoạn chuyển mùa; lượng mưa ít, độ lệch chuẩn lượng mưa chỉ là $\sigma_{\text{val}} = 1.34\text{ mm/h}$.
* Tập Test (H2/2025 - Sáu tháng cuối năm): Rơi vào **cao điểm mùa mưa Tây Nam tại Nam Bộ** với độ lệch chuẩn lượng mưa tăng vọt lên **$\sigma_{\text{test}} = 1.91\text{ mm/h}$ (+42.6%)**, kèm theo các đợt mưa dông cực đoan lịch sử vào tháng 10/2025.

Khi phương sai tự nhiên của chuỗi thời gian tăng hơn 40%, sai số tuyệt đối tăng theo là quy luật khách quan, chứng minh mô hình tổng quát hóa tốt trên phân phối mùa mưa thực tế.

![Thực tế vs Dự báo trên Test](results/figures/actual_vs_predicted_test.png)

---

## 7. ĐẢM BẢO CHẤT LƯỢNG KỸ THUẬT PHẦN MỀM (MLOPS & VERIFICATION)

### 7.1. Kiểm thử Tính tương đương Tuyệt đối (Parity Test Against Train-Serving Skew)
Trong công nghệ MLOps, lỗi nghiêm trọng nhất là độ lệch giữa lúc huấn luyện (Batch Pandas) và lúc triển khai (Single-Sample NumPy Array). Nhóm đã thiết kế ca kiểm thử tự động `test_feature_parity_batch_vs_single_sample` trong [test_pipeline.py](file:///d:/rain-latency-optimization/tests/test_pipeline.py):
* Kiểm tra tương đồng vector đặc trưng trên toàn bộ các mức $L \in \{1, 6, 24, 72\}$: sai lệch vector $\le 1.9 \times 10^{-6}$;
* Kiểm tra đầu ra dự báo của Champion Model trên 25 mẫu ngẫu nhiên: **độ chênh lệch dự báo $< 10^{-5}$** (thực tế đo được $0.0$ ở mức làm tròn 5 chữ số thập phân, triệt tiêu hoàn toàn sai lệch số học).

### 7.2. Tích hợp Liên tục (CI/CD) trên GitHub Actions với Python 3.13
Repository được cấu hình quy trình CI tự động ([.github/workflows/ci.yml](file:///d:/rain-latency-optimization/.github/workflows/ci.yml)) trên môi trường Ubuntu với Python 3.13. Mỗi commit được tự động kiểm toán dependency, tải dữ liệu Meteostat, chạy 6 ca unit test nghiêm ngặt và kiểm tra demo suy luận. Trạng thái hiện tại: **100% Passed (Green ✔)**.

### 7.3. Triển khai Ứng dụng Web Dashboard Tương tác Trực quan
Nhóm đã xây dựng một Web Application hoàn chỉnh (`src/server.py` và thư mục `web/`), cung cấp giao diện trực quan tại `http://127.0.0.1:8000`, hỗ trợ:
* Mô phỏng thời tiết tức thời (Nhiệt độ, Độ ẩm, Áp suất, Mưa giờ trước);
* Đo đạc độ trễ phần cứng thực tế (Hardware Latency Benchmark qua $N=200$ lần chạy);
* Đối sánh thời gian thực 7 mô hình tham chiếu và phát cảnh báo cấp độ ngập lụt.

---

## 8. GIỚI HẠN NGHIÊN CỨU & ĐỊNH HƯỚNG PHÁT TRIỂN
1. **Độ lệch Mẫu do Cắt bỏ Giá trị Khuyết theo $L$ (Sample Truncation by Lag):** Mỗi giá trị $L$ đòi hỏi $L$ hàng dữ liệu quá khứ liên tục, dẫn tới việc cấu hình $L=72$ mất 71 dòng dữ liệu đầu tiên nhiều hơn cấu hình $L=1$ khi `dropna`. Mặc dù độ lệch này rất nhỏ trên tổng số 17.286 mẫu train (~0.4%), định hướng tiếp theo sẽ chuẩn hóa bằng cách cắt bỏ cố định $L_{\max} = 72$ dòng đầu tiên cho tất cả các cấu hình để đảm bảo tập mẫu thẩm định hoàn toàn bất biến tuyệt đối.
2. **Chiến lược Kiểm định Chéo Theo Chuỗi Thời Gian (Rolling-Origin Cross-Validation):** Do tính chất phân kỳ mùa rõ rệt, nghiên cứu hiện tại áp dụng phân chia niên lịch cố định. Hướng phát triển tiếp theo là áp dụng Time-Series Rolling-Origin Cross-Validation qua nhiều chu kỳ năm nhằm kiểm chứng tính bền vững của các siêu tham số trước các hiện tượng El Niño / La Niña bất thường.
3. **Mở rộng Mạng lưới Quan trắc Đa trạm (Spatial-Temporal Modeling):** Mở rộng bài toán từ trạm đơn Tân Sơn Nhất sang mạng lưới các trạm lân cận (Nhà Bè, Cần Giờ, Bình Triệu) để mô hình hóa tính dị thể không gian của hoàn lưu gió mùa.

---

## 9. KẾT LUẬN & ĐÓNG GÓP CỦA ĐỒ ÁN
Nghiên cứu đã giải quyết thành công bài toán tối ưu hóa siêu tham số và chiều sâu đặc trưng có ràng buộc độ trễ thời gian thực cho dự báo lượng mưa ngắn hạn tại trạm Tân Sơn Nhất. Các đóng góp chính gồm:
1. **Thiết lập bài toán tối ưu hóa có ràng buộc chặt chẽ:** Phối hợp giữa $L$ và $\theta$ với ràng buộc $T_{\text{e2e}} \le 5.0\text{ ms}$ (biên an toàn $\le 4.5\text{ ms}$).
2. **Minh chứng ưu thế của Optuna TPE:** Đạt tỷ lệ khả thi vượt trội (+14.0 pp) và giảm phương sai thực nghiệm 57.6% qua 5 random seeds.
3. **Khám phá kiến trúc tinh gọn $L=1$:** Giảm 77.8% số chiều đặc trưng và giảm 83.3% số hàng bộ nhớ đệm chuỗi thời gian, đạt Test RMSE 1.1774 mm/h, $F_1 = 0.5882$, độ trễ C-API 1.04 ms.
4. **Đảm bảo tính tái lập chuẩn mực:** Hệ thống mã nguồn mở hoàn chỉnh, 100% unit tests pass, đi kèm Web Dashboard tương tác thực tế.

---

## TÀI LIỆU THAM KHẢO (REFERENCES)

1. J. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, "Algorithms for hyper-parameter optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 24, 2011.
2. T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in *Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, 2019, pp. 2623–2631.
3. G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
4. Meteostat Open Weather and Climate Data, "Hourly weather data for weather station Tan Son Nhat (ID: 48900)," *Meteostat Python Package*, 2026. [Online]. Available: `https://meteostat.net/`.
5. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
