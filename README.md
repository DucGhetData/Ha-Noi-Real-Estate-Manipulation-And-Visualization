# 🏡 Real Estate Data Crawler And Visualization - Hanoi, Vietnam

## 📖 Mô tả dự án
Trong bối cảnh những năm gần đây giá bất động sản ở Việt Nam tăng cao, việc có thể nhìn nhận tình hình bất động sản là vô cùng quan trọng trong quyết định mua nhà nhằm đầu tư hay sinh sống. Nắm được nhu cầu đó, dự án này sẽ sử dụng dữ liệu bất động sản tại Việt Nam những năm 2019-2020, thời điểm bất động sản ở các thành phố lớn Hà Nội gần như đóng băng, nhằm đưa ra những góc nhìn tổng quan nhất về giá, diện tích, số lượng giao bán hàng tháng,...

## 🎯 Mục tiêu
- Làm sạch và chuẩn hóa dữ liệu.
- Phân tích xu hướng giá, khu vực, loại hình bất động sản
- Trực quan hóa trên Tableau phục vụ báo cáo hoặc ra quyết định đầu tư
- Sử dụng các kỹ thuật trong Machine Learning nhằm xây dựng một mô hình đơn giản để dự đoán giá nhà

## 📅 Quy trình thực hiện dự án
1. Dự án sẽ sử dụng nguồn dữ liệu từ bên thứ 3 cung cấp, đường dẫn:https://www.kaggle.com/datasets/ladcva/vietnam-housing-dataset-hanoi
2. Sử dụng Pandas và các thư viện bổ trợ khác để làm sạch và chuẩn hóa dữ liệu. Các thông tin không cần thiết hoặc sinh ra do quá trình trích xuất dữ liệu sẽ bị loại bỏ, các thông tin cần thiết được điều chỉnh để phù hợp cho việc trực quan hóa. Dữ liệu tiếp tục được lưu trong file .csv
3. Tiếp theo là sử dụng Tableau để trực quan hóa dữ liệu và đưa ra các kết luận và phân tích về giá bất động sản ở Hà Nội
4. Cuối cùng là sử dụng thư viện scikit-learn và các kỹ thuật Random search và Grid search để xây dựng mô hình dự đoán giá nhà

## 📊 Báo cáo
- Xem chi tiết báo cáo tại đây: [Ha Noi Real Estate 2019-2020 Report](https://public.tableau.com/views/HaNoiRealEstateReport/BocotnhhnhBSHNi?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) 

## 📂 Cấu trúc thư mục
```bash
Ha Noi Real Estate Visualization/
├── data/                # Chứa file dữ liệu gốc và dữ liệu đã xử lý
├── notebooks/           # Jupyter Notebooks (xử lý, phân tích, trực quan)
├── tableau/             # File tableau
├── src/                 # Mã nguồn Python (các script, xử lý, xây dựng mô hình,...)
├── README.md            # File mô tả dự án
├── requirements.txt     # các thư viện cần cài đặt