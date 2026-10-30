# ĐỀ XUẤT ĐỀ TÀI

## Tên đề tài

**Tối ưu hóa mô hình dự báo lượng mưa một giờ tới dưới ràng buộc độ trễ suy luận**

**English title:**  
**Optimization of Next-Hour Rainfall Forecasting under an End-to-End Inference Latency Constraint**

## 1. Bối cảnh và vấn đề nghiên cứu

Dự báo lượng mưa trong thời gian ngắn có ý nghĩa quan trọng đối với các hệ thống cảnh báo mưa lớn và ngập đô thị. Trong các ứng dụng thời gian thực, một mô hình dự báo không chỉ cần có sai số thấp mà còn phải đáp ứng giới hạn về thời gian xử lý trên phần cứng triển khai.

Các mô hình học máy có độ phức tạp cao có thể cải thiện độ chính xác nhưng đồng thời làm tăng thời gian trích xuất đặc trưng và suy luận. Vì vậy, việc lựa chọn mô hình chỉ dựa trên độ chính xác có thể tạo ra một cấu hình không phù hợp với yêu cầu vận hành thực tế.

Đề tài xem bài toán dự báo lượng mưa một giờ tiếp theo như một bài toán tối ưu hóa có ràng buộc, trong đó đồng thời lựa chọn độ dài cửa sổ dữ liệu lịch sử và các siêu tham số của mô hình LightGBM nhằm giảm sai số dự báo nhưng vẫn đảm bảo giới hạn độ trễ suy luận trên CPU.

## 2. Câu hỏi nghiên cứu

Với dữ liệu khí tượng lịch sử theo giờ, cần lựa chọn độ dài cửa sổ lịch sử \(L\) và cấu hình siêu tham số \(\theta\) như thế nào để đạt sai số dự báo lượng mưa một giờ tới nhỏ nhất trong khi vẫn đảm bảo độ trễ suy luận end-to-end không vượt quá 5 ms trên CPU?

Ngoài ra, đề tài khảo sát liệu phương pháp tối ưu hóa Bayesian sử dụng Optuna TPE có khả năng tìm được nghiệm khả thi và hội tụ đến cấu hình tốt hơn Random Search khi hai phương pháp được cung cấp cùng ngân sách đánh giá hay không.

## 3. Mục tiêu nghiên cứu

### Mục tiêu tổng quát

Xây dựng và đánh giá một quy trình tối ưu hóa mô hình dự báo lượng mưa ngắn hạn có xét đồng thời độ chính xác và giới hạn độ trễ suy luận.

### Mục tiêu cụ thể

1. Thu thập và tiền xử lý dữ liệu khí tượng theo giờ của trạm Tân Sơn Nhất.

2. Xây dựng tập đặc trưng nhân quả phục vụ dự báo lượng mưa tại thời điểm \(t+1\), đảm bảo không sử dụng thông tin từ tương lai.

3. Xây dựng các mô hình baseline gồm Persistence, Linear Regression, Ridge Regression và LightGBM mặc định.

4. Xây dựng bài toán tối ưu hóa đồng thời độ dài cửa sổ lịch sử \(L\) và siêu tham số LightGBM.

5. So sánh Random Search và Optuna TPE dưới cùng ngân sách đánh giá.

6. Đánh giá sự đánh đổi giữa độ chính xác dự báo và độ trễ suy luận.

7. Kiểm tra khả năng tổng quát hóa của nghiệm tối ưu trên tập Test hoàn toàn độc lập.

## 4. Dữ liệu

Nguồn dữ liệu chính là dữ liệu khí tượng lịch sử theo giờ từ Meteostat tại trạm Tân Sơn Nhất, mã trạm WMO 48900.

Khoảng dữ liệu phục vụ mô hình hóa chính là giai đoạn 2023–2025.

Các biến khí tượng được sử dụng gồm:

- nhiệt độ;
- điểm sương;
- độ ẩm tương đối;
- áp suất;
- tốc độ gió;
- lượng mưa;
- các biến lịch sử và thống kê trượt được xây dựng từ dữ liệu quá khứ.

Biến mục tiêu là lượng mưa tại một giờ tiếp theo:

\[
y_{t+1}=\text{precipitation at time }t+1.
\]

Dữ liệu được chia theo thời gian thay vì lấy mẫu ngẫu nhiên nhằm tránh rò rỉ dữ liệu.

Tập Train sử dụng dữ liệu đến cuối năm 2024.

Tập Validation sử dụng sáu tháng đầu năm 2025 và được dùng trong quá trình tối ưu hóa.

Tập Test sử dụng sáu tháng cuối năm 2025 và chỉ được sử dụng sau khi quá trình lựa chọn cấu hình đã hoàn tất.

## 5. Mô hình hóa bài toán tối ưu

Vector biến quyết định được xác định bởi:

\[
x=(L,\theta)
\]

trong đó \(L\) là độ dài cửa sổ lịch sử và \(\theta\) là tập siêu tham số của LightGBM.

Các giá trị cửa sổ được khảo sát:

\[
L\in\{1,2,3,6,12,24,36,48,72\}.
\]

Một số siêu tham số chính gồm:

\[
\theta=
(
n_{\text{estimators}},
max_{\text{depth}},
num_{\text{leaves}},
learning_{\text{rate}},
min_{\text{child samples}}
).
\]

Hàm mục tiêu là:

\[
\min_{L,\theta}
RMSE_{\text{validation}}(L,\theta).
\]

Với ràng buộc:

\[
T_{e2e}(L,\theta)\le5.0\text{ ms}.
\]

Độ trễ end-to-end được định nghĩa:

\[
T_{e2e}
=
T_{\text{feature}}(L)
+
T_{\text{predict}}(\theta).
\]

Giá trị \(P_{95}\) của phân phối thời gian suy luận được sử dụng thay vì thời gian trung bình nhằm hạn chế ảnh hưởng của nhiễu đo và phản ánh tốt hơn yêu cầu hệ thống thời gian thực.

Một cấu hình được xem là khả thi khi:

\[
P_{95}(T_{e2e})\le5.0\text{ ms}.
\]

## 6. Phương pháp tối ưu hóa

Hai phương pháp được sử dụng.

### Random Search

Random Search đóng vai trò baseline. Các cấu hình được lấy mẫu ngẫu nhiên trong cùng không gian tìm kiếm với phương pháp chính.

### Optuna TPE

Tree-structured Parzen Estimator được sử dụng như phương pháp tối ưu hóa chính. TPE sử dụng kết quả của các trial trước để ưu tiên các vùng của không gian tìm kiếm có khả năng tạo ra nghiệm tốt.

Hai phương pháp được đánh giá với cùng số trial và cùng các random seed để đảm bảo tính công bằng.

Thí nghiệm chính sử dụng 40 trial trên mỗi seed để so sánh trực tiếp Random Search và Optuna TPE.

Một thí nghiệm mở rộng với ngân sách lớn hơn được thực hiện cho Optuna nhằm khảo sát đường cong hội tụ và đánh giá liệu việc tăng số trial có tiếp tục cải thiện nghiệm hay không.

## 7. Chiến lược đo độ trễ

Để giảm chi phí tính toán trong quá trình tối ưu hóa, việc đo latency được thực hiện theo hai giai đoạn.

### Giai đoạn tìm kiếm

Mỗi trial sử dụng một benchmark ngắn để ước lượng latency và loại các cấu hình có khả năng vi phạm giới hạn.

Chi phí trích xuất đặc trưng phụ thuộc chủ yếu vào \(L\), vì vậy nó được benchmark trước cho từng giá trị cửa sổ và tái sử dụng trong quá trình tối ưu hóa.

### Giai đoạn xác nhận

Sau khi quá trình tìm kiếm hoàn tất, các nghiệm khả thi tốt nhất được benchmark lại với số lần lặp lớn hơn.

Benchmark chính thức sử dụng warm-up trước khi đo và tối thiểu 200 lần lặp cho mỗi cấu hình.

Các chỉ số latency được báo cáo gồm:

\[
P_{50},\quad P_{95},\quad P_{99}.
\]

Giá trị \(P_{95}\) được sử dụng để kiểm tra điều kiện khả thi.

## 8. Chỉ số đánh giá

Các chỉ số dự báo chính gồm:

- Validation RMSE;
- Test RMSE;
- Test MAE;
- Wet-hour RMSE;
- Rain-event Precision;
- Rain-event Recall;
- Rain-event \(F_1\).

Các chỉ số về hiệu năng tính toán gồm:

- \(P_{50}\) latency;
- \(P_{95}\) latency;
- \(P_{99}\) latency;
- tỷ lệ nghiệm khả thi;
- số trial cần thiết để đạt best-so-far RMSE.

Đường cong hội tụ và biên accuracy–latency được sử dụng để trực quan hóa hành vi của hai phương pháp tối ưu hóa.

## 9. Thực nghiệm

Các nhóm mô hình được so sánh gồm:

1. Persistence baseline;
2. Linear Regression;
3. Ridge Regression;
4. LightGBM mặc định;
5. LightGBM tối ưu bằng Random Search;
6. LightGBM tối ưu bằng Optuna TPE.

Một phát hiện thực nghiệm quan trọng cần được kiểm chứng là mô hình LightGBM mặc định có thể vượt giới hạn latency 5 ms khi benchmark bằng quy trình đo có độ chính xác cao.

Điều này minh họa rằng cấu hình có độ chính xác tốt không mặc nhiên phù hợp cho triển khai thời gian thực và tạo động lực trực tiếp cho bài toán tối ưu hóa có ràng buộc.

## 10. Kết quả kỳ vọng

Đề tài kỳ vọng tìm được một cấu hình LightGBM có sai số dự báo thấp đồng thời thỏa mãn giới hạn:

\[
P_{95}\le5.0\text{ ms}.
\]

Ngoài việc tìm cấu hình tốt nhất, nghiên cứu hướng tới phân tích:

- ảnh hưởng của cửa sổ lịch sử đến độ chính xác và latency;
- ảnh hưởng của độ phức tạp mô hình đến latency;
- khả năng tìm nghiệm khả thi của Random Search và TPE;
- tốc độ hội tụ của hai phương pháp;
- sự đánh đổi giữa accuracy và latency.

Kết quả cuối cùng không chỉ là một mô hình dự báo lượng mưa mà còn là một quy trình lựa chọn mô hình dưới giới hạn tài nguyên tính toán có thể tái lập trên phần cứng thực tế.

## 11. Giới hạn nghiên cứu

Dữ liệu hiện chỉ được lấy tại một trạm khí tượng nên chưa mô hình hóa được sự biến thiên không gian của lượng mưa trên toàn TP.HCM.

Latency phụ thuộc vào phần cứng, hệ điều hành, phiên bản thư viện và tải của CPU tại thời điểm benchmark.

Dữ liệu lượng mưa có phân phối mất cân bằng với tỷ lệ giờ không mưa lớn, do đó RMSE tổng thể chưa thể hiện đầy đủ chất lượng dự báo các sự kiện mưa lớn.

Các hạn chế trên được xử lý một phần thông qua việc báo cáo metadata phần cứng, sử dụng \(P_{95}\) latency và bổ sung các chỉ số riêng cho các giờ có mưa.

## 12. Đóng góp dự kiến

Đóng góp chính của đề tài là xây dựng một bài toán tối ưu hóa mô hình học máy có ràng buộc hiệu năng triển khai thực tế.

Thay vì lựa chọn mô hình chỉ dựa trên độ chính xác, đề tài tích hợp trực tiếp độ trễ end-to-end vào quá trình lựa chọn cấu hình.

Qua đó, đề tài làm rõ sự đánh đổi giữa chất lượng dự báo và chi phí suy luận, đồng thời đánh giá khả năng của Bayesian Optimization trong việc tìm kiếm nghiệm tốt dưới một ngân sách tính toán hữu hạn.
