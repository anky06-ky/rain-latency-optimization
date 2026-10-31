# ĐỀ XUẤT ĐỀ TÀI

## Tên đề tài

**Tối ưu hóa mô hình dự báo lượng mưa một giờ tới dưới ràng buộc độ trễ suy luận**

**English title:**  
**Optimization of Next-Hour Rainfall Forecasting under an End-to-End Inference Latency Constraint**

## 1. Bối cảnh và vấn đề nghiên cứu

Dự báo lượng mưa trong thời gian ngắn (short-term rainfall nowcasting) có ý nghĩa quyết định đối với các hệ thống cảnh báo sớm ngập lụt đô thị và điều hành trạm bơm tự động. Trong các kịch bản triển khai cạnh (Edge IoT) hoặc vòng quét thời gian thực liên tục qua mạng lưới trạm quan trắc, một mô hình dự báo không chỉ cần sai số thấp mà bắt buộc phải thỏa mãn nghiêm ngặt giới hạn về thời gian xử lý phần cứng.

Các mô hình học máy phức tạp có thể tăng nhẹ độ chính xác nhưng lại kéo dài đáng kể thời gian trích xuất đặc trưng và suy luận cây quyết định. Việc lựa chọn mô hình chỉ dựa trên độ chính xác thuần túy sẽ dễ dẫn tới những cấu hình "quá tải phần cứng", không khả thi trong vận hành thực tế.

Đề tài tiếp cận bài toán dự báo lượng mưa một giờ tiếp theo dưới góc nhìn của **Bài toán Tối ưu hóa có Ràng buộc (Constrained Optimization)**: Đồng thời tìm kiếm độ dài cửa sổ lịch sử \(L\) và bộ siêu tham số \(\theta\) của mô hình LightGBM nhằm cực tiểu hóa sai số dự báo trên tập thẩm định trong khi bảo đảm độ trễ suy luận toàn trình không vượt quá 5.0 ms trên CPU đơn luồng.

## 2. Câu hỏi nghiên cứu (Research Questions)

Đề tài tập trung giải quyết ba câu hỏi nghiên cứu then chốt:

* **RQ1 (Hiệu quả dưới cùng ngân sách đánh giá - Fair Comparison under Equal Budget):**  
  *Dưới cùng một ngân sách đánh giá hạn chế (\(N = 40\) trials/seed trên cùng 5 random seeds), liệu phương pháp tối ưu hóa Bayesian sử dụng Optuna TPE có tìm được cấu hình thỏa mãn ràng buộc độ trễ \(P_{95} \le 5.0\text{ ms}\) với sai số \(RMSE_{\text{val}}\) thấp hơn và tỷ lệ nghiệm khả thi (feasible rate) vượt trội so với Random Search hay không?*

* **RQ2 (Mở rộng ngân sách & Tính bão hòa hội tụ - Budget Scaling & Convergence Saturation):**  
  *40 trials đóng vai trò đối chuẩn công bằng, không phải giới hạn của dự án. Khi mở rộng ngân sách tối ưu hóa lên quy mô lớn hơn (\(N = 100\) trials/seed \(\times\) 5 seeds = 500 trials), nghiệm tối ưu của Optuna TPE có tiếp tục được cải thiện hay không? Đường cong hội tụ thể hiện điểm bão hòa (diminishing returns) tại ngưỡng ngân sách nào?*

* **RQ3 (Đánh đổi Động lực học & Biên Pareto - Accuracy-Latency Trade-off):**  
  *Khi mở rộng không gian tìm kiếm số lượng cây \(n_{\text{estimators}} \in [20, 500]\), sự đánh đổi giữa độ phức tạp mô hình, độ dài cửa sổ lịch sử \(L\) và thời gian suy luận toàn trình \(T_{e2e}\) diễn ra như thế nào trên đường biên Pareto?*

## 3. Mục tiêu nghiên cứu

### Mục tiêu tổng quát

Xây dựng và đánh giá một quy trình tối ưu hóa mô hình dự báo lượng mưa ngắn hạn có xét đồng thời độ chính xác và giới hạn độ trễ suy luận thực tế trên CPU đơn luồng.

### Mục tiêu cụ thể

1. **Thu thập & tiền xử lý chuẩn mực:** Thu thập dữ liệu khí tượng theo giờ trạm Tân Sơn Nhất (WMO ID: 48900) giai đoạn 2023–2025; áp dụng điền khuyết nhân quả (causal forward-fill) triệt tiêu 100% rò rỉ dữ liệu (data leakage).
2. **Xây dựng đặc trưng nhân quả:** Thiết lập không gian đặc trưng gồm trễ quan sát và thống kê trượt theo các cửa sổ \(L\), đảm bảo không sử dụng bất kỳ thông tin nào từ tương lai.
3. **Xây dựng baseline tham chiếu:** Thiết lập 4 mô hình cơ sở gồm Persistence baseline, Linear Regression, Ridge Regression và LightGBM mặc định.
4. **Mô hình hóa bài toán tối ưu có ràng buộc:** Định nghĩa chính xác vector biến quyết định \(x = (L, \theta)\), hàm mục tiêu \(\min RMSE_{\text{val}}\), và ràng buộc phân vị \(P_{95}(T_{e2e}) \le 5.0\text{ ms}\).
5. **Đánh giá đối chuẩn 2 cấp độ (Two-Tier Optimization Framework):**
   - *Cấp độ 1 (RQ1):* Đối chuẩn công bằng Random Search vs Optuna TPE tại ngân sách 40 trials/seed qua 5 seeds.
   - *Cấp độ 2 (RQ2):* Mở rộng ngân sách Optuna TPE lên 500 trials để khảo sát động lực học hội tụ và tiệm cận đường biên Pareto tối ưu.
6. **Khảo sát đánh đổi & khả năng tổng quát hóa:** Phân tích đường biên Pareto giữa RMSE và latency, kiểm chứng nghiệm tối ưu trên tập Test độc lập hoàn toàn.

## 4. Dữ liệu nghiên cứu

* **Nguồn dữ liệu:** Chuỗi dữ liệu khí tượng theo giờ từ Meteostat tại trạm Tân Sơn Nhất (WMO ID: 48900), giai đoạn 2023–2025 gồm 26.304 mốc thời gian liên tục.
* **Các biến khí tượng quan trắc:**
  - Nhiệt độ không khí (`temp`, °C);
  - Điểm sương (`dwpt`, °C);
  - Độ ẩm tương đối (`rhum`, %);
  - Áp suất khí quyển (`pres`, hPa);
  - Tốc độ gió (`wspd`, km/h);
  - Lượng mưa quan trắc (`prcp`, mm).
* **Biến mục tiêu:** Lượng mưa tại 1 giờ tiếp theo:
  \[
  y_{t+1} = \text{precipitation at time } t+1.
  \]
* **Phân chia tập dữ liệu chặt chẽ theo dòng thời gian (Chronological Split):**
  - **Tập Train (70%):** Dữ liệu từ 01/01/2023 đến 31/12/2024 (2 năm đầu).
  - **Tập Validation (15%):** Sáu tháng đầu năm 2025 (01/01/2025 – 30/06/2025), dùng độc quyền để tính toán hàm mục tiêu trong quá trình tối ưu hóa.
  - **Tập Test (15%):** Sáu tháng cuối năm 2025 (01/07/2025 – 31/12/2025), khóa kín và chỉ đánh giá một lần duy nhất sau khi đã chốt cấu hình tối ưu.

## 5. Mô hình hóa bài toán tối ưu

Vector biến quyết định được chuẩn hóa:

\[
x = (L, \theta)
\]

trong đó:
- \(L\) là độ dài cửa sổ lịch sử quan sát:
  \[
  L \in \{1, 2, 3, 6, 12, 24, 36, 48, 72\} \text{ (giờ)}.
  \]
- \(\theta\) là tập 5 siêu tham số cấu trúc của mô hình LightGBM:
  \[
  \theta = (n_{\text{estimators}}, \max_{\text{depth}}, \text{num}_{\text{leaves}}, \text{learning}_{\text{rate}}, \text{min}_{\text{child\_samples}})
  \]
  Không gian tìm kiếm:
  - \(n_{\text{estimators}} \in [20, 500]\) *(mở rộng lên 500 cây nhằm khảo sát giới hạn chịu tải latency)*;
  - \(\max_{\text{depth}} \in [2, 12]\);
  - \(\text{num}_{\text{leaves}} \in [7, 127]\);
  - \(\text{learning}_{\text{rate}} \in [0.01, 0.25]\) *(lấy mẫu log-uniform)*;
  - \(\text{min}_{\text{child\_samples}} \in [10, 100]\).

### Hàm mục tiêu

\[
\min_{L, \theta} \quad f(L, \theta) = RMSE_{\text{validation}}(L, \theta)
\]

### Ràng buộc độ trễ phần cứng

\[
T_{e2e}(L, \theta) = T_{\text{feature}}(L) + T_{\text{predict}}(\theta) \le 5.0\text{ ms}
\]

Điều kiện khả thi được kiểm định thông qua phân vị 95 của phân phối đo lường:

\[
P_{95}(T_{e2e}) \le 5.0\text{ ms}
\]

## 6. Phương pháp tối ưu hóa & Thiết kế thực nghiệm

### 6.1. Các thuật toán tối ưu hóa
1. **Random Search (Baseline):** Lấy mẫu ngẫu nhiên đồng đều trong không gian tìm kiếm. Sử dụng phương pháp lọc hậu nghiệm (post-hoc filtering) để loại bỏ các trial vi phạm ràng buộc \(P_{95} > 5.0\text{ ms}\).
2. **Optuna TPE (Phương pháp đề xuất chính):** Tree-structured Parzen Estimator ứng dụng mô hình hóa Bayesian. TPE tích hợp hàm ràng buộc mềm để ước lượng phân phối xác suất nghiệm khả thi, ưu tiên khám phá các vùng không gian vừa giảm RMSE vừa chạy dưới 5.0 ms.

### 6.2. Thiết kế thực nghiệm phân tầng (Two-Tier Experimental Setup)
* **Pha 1: Đối chuẩn công bằng (Fair Comparison Benchmark - RQ1):**
  - Chạy 40 trials/seed trên 5 bộ random seed độc lập cho cả Random Search và Optuna TPE.
  - So sánh trực tiếp: Best-so-far RMSE, tỷ lệ nghiệm khả thi (feasible rate %), và tốc độ hội tụ ban đầu dưới cùng ngân sách tính toán.
* **Pha 2: Khảo sát mở rộng ngân sách (Budget Scaling Analysis - RQ2):**
  - 40 trials chỉ là mức ngân sách đối chuẩn, không phải giới hạn của đồ án.
  - Mở rộng ngân sách Optuna TPE lên quy mô 100 trials/seed \(\times\) 5 seeds = **500 trials**.
  - Khảo sát sự dịch chuyển của đường biên Pareto, kiểm tra xem liệu việc tăng ngân sách có tiếp tục cải thiện sai số hay đạt điểm bão hòa, đồng thời ghi nhận nghiệm tốt nhất toàn cục (Champion Model).

## 7. Giao thức đo độ trễ suy luận (Latency Protocol)

Để bảo đảm độ tin cậy và hạn chế tối đa nhiễu đo lường trên hệ điều hành:
1. **Đo toàn trình (End-to-End):** Bao gồm thời gian trích xuất đặc trưng đơn mẫu từ cửa sổ \(L\) cộng với thời gian chạy cây quyết định trên CPU đơn luồng (`n_jobs=1`), ứng dụng API gọi trực tiếp `booster_.predict` để loại trừ hoàn toàn chi phí overhead của wrapper Python.
2. **Khởi động CPU (Warm-up):** Thực hiện 20 lần chạy suy luận giả lập trước khi đo để nạp sẵn cache bộ nhớ và ổn định tiến trình.
3. **Đo lường phân vị thống kê & Biên an toàn:** Đo lặp 100 lần bằng đồng hồ độ chính xác nanô-giây (`time.perf_counter_ns()`), xuất ra các phân vị \(P_{50}, P_{95}, P_{99}\). Ngoài ràng buộc SLA $5.0\text{ ms}$, đề tài thiết lập **biên an toàn nội bộ $4.5\text{ ms}$** để phòng ngừa nhiễu điều tiết xung nhịp CPU.

## 8. Chỉ số đánh giá toàn diện

* **Chỉ số chất lượng dự báo:**
  - $RMSE_{\text{val}}$, $RMSE_{\text{test}}$, $MAE_{\text{test}}$;
  - $RMSE$ trên các giờ thực sự có mưa ($Wet\_Test\_RMSE$);
  - Precision, Recall, $F_1$-score phát hiện sự kiện mưa ($prcp \ge 0.5$ mm);
  - Chỉ số kỹ năng dự báo ($Skill\_Score$) đối chuẩn với Persistence và Climatology.
* **Chỉ số hiệu năng tối ưu & tính toán:**
  - Phân vị độ trễ $P_{50}, P_{95}, P_{99}$ (ms);
  - Tỷ lệ nghiệm khả thi (Feasible Rate %);
  - Số trial cần thiết để đạt điểm hội tụ nghiệm tốt nhất;
  - Đường cong hội tụ sai số theo số trial và biên Pareto Accuracy–Latency.

## 9. Cấu trúc thực nghiệm so sánh

Hệ thống tiến hành so sánh đối chuẩn 7 cấu hình mô hình:
1. **Persistence Baseline:** Dự báo lượng mưa giờ tới bằng đúng lượng mưa giờ hiện tại.
2. **Climatology Baseline:** Dự báo lượng mưa dựa trên kỳ vọng lịch sử theo tháng và khung giờ trong ngày $\mathbb{E}[y \mid \text{month}, \text{hour}]$.
3. **Linear Regression:** Mô hình hồi quy tuyến tính chuẩn (OLS).
4. **Ridge Regression:** Hồi quy tuyến tính có điều chuẩn L2.
5. **LightGBM Mặc định:** Cấu hình mặc định của thư viện (100 cây, depth tự do).
6. **LightGBM (Tối ưu bởi Random Search):** Nghiệm tốt nhất thỏa mãn ràng buộc.
7. **LightGBM (Tối ưu bởi Optuna TPE):** Nghiệm tối ưu toàn cục đề xuất ($L=1$, 150 cây, depth=12, lr=0.0215).

## 10. Kết quả kỳ vọng & Đóng góp của đề tài

1. **Minh chứng về sự cần thiết của tối ưu hóa có ràng buộc:** Chứng minh mô hình LightGBM mặc định hoặc các cấu hình quá nhiều cây ($n_{\text{estimators}} > 300$) dễ bị vượt ngưỡng 5.0 ms, khẳng định việc chỉ nhìn vào độ chính xác là không đủ cho ứng dụng thời gian thực.
2. **Vượt trội của Bayesian Optimization:** Chứng minh Optuna TPE đạt tỷ lệ nghiệm khả thi cao hơn đáng kể so với Random Search dưới cùng ngân sách (RQ1) và tiếp tục cải thiện chất lượng nghiệm khi mở rộng ngân sách lên 500 trials (RQ2).
3. **Lời giải trọn vẹn cho bài toán thực tế:** Cung cấp cấu hình tối ưu toàn diện (độ dài cửa sổ $L$ và bộ siêu tham số $\theta$) kèm quy trình tái lập trọn vẹn từ dữ liệu thô đến triển khai suy luận.
