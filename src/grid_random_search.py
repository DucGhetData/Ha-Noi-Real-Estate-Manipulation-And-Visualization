import time
import pandas as pd   # Pandas để cleaning data
import numpy as np
import category_encoders as ce # Thư viện hỗ trợ Target Encoding
import joblib
import threading
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split # Thư viện hỗ trợ chia dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("F:\\Data Analyst\\Ha Noi Real Estate Visualization\\data\\VN_housing_dataset_cleaned.csv")

# Hàm xử lý dữ liệu
def preprocess_data(df):
    # Loại bỏ các dòng không cần thiết
    df = df.drop(columns=['Địa chỉ', 'Quận_tableau', 'Tỉnh/Thành phố'])

    # Đánh index cho ngày giúp mô hình hiểu được biến động theo thời gian
    df['Ngày'] = pd.to_datetime(df['Ngày'])
    df['Ngày_index'] = (df['Ngày'] - df['Ngày'].min()).dt.days
    df = df.drop(columns=['Ngày'])
    return df

df = preprocess_data(df)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử (80% huấn luyện, 20% kiểm thử)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
# Thực hiện One-Hot Encoding cho 2 cột: "Loại hình nhà ở" và "Giấy tờ pháp lý"
df_train = pd.get_dummies(df_train, columns=['Loại hình nhà ở', 'Giấy tờ pháp lý'], drop_first=True)

# Làm với df_test thì tương tự
df_test = pd.get_dummies(df_test, columns=['Loại hình nhà ở', 'Giấy tờ pháp lý'], drop_first=True)
    
# Khởi tạo Target Encoder
target_encoder = ce.TargetEncoder(cols=['Quận', 'Huyện'])

# Áp dụng Target Encoding trên tập huấn luyện
df_train[['Quận_encoded', 'Huyện_encoded']] = target_encoder.fit_transform(df_train[['Quận', 'Huyện']], df_train['Giá (triệu đồng/m2)'])

# Áp dụng Target Encoding trên tập kiểm thử, nhưng chỉ dùng transform (không fit lại)
df_test[['Quận_encoded', 'Huyện_encoded']] = target_encoder.transform(df_test[['Quận', 'Huyện']])

# Chia tập huấn luyện
X_train = df_train.drop('Giá (triệu đồng/m2)', axis=1)  # Các đặc trưng
y_train = df_train['Giá (triệu đồng/m2)']  # Biến mục tiêu

# Chia tập kiểm thử
X_test = df_test.drop('Giá (triệu đồng/m2)', axis=1)  # Các đặc trưng
y_test = df_test['Giá (triệu đồng/m2)']  # Biến mục tiêu

# Loại bỏ cột "Quận" và "Huyện" gốc (chưa mã hóa)
X_train = X_train.drop(columns=['Quận', 'Huyện'])
X_test = X_test.drop(columns=['Quận', 'Huyện'])


# Hàm để đếm thời gian trong khi huấn luyện
training_done = False
def display_training_time():
    start_time = time.time()
    while not training_done:
        elapsed_time = time.time() - start_time
        print(f"Mô hình đang huấn luyện: {int(elapsed_time)} giây", end="\r")
        time.sleep(1)
        
# Bắt đầu luồng đếm thời gian
thread = threading.Thread(target=display_training_time)
thread.start()  

# Tạo thư viện chứa các giá trị ngẫu nhiên của mô hình
RF_random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
               'max_features': ['auto', 'sqrt', 'log2'],
               'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}

# Tạo mô hình RF cơ sở và phù hợp với tìm kiếm ngẫu nhiên
RF_regressor = RandomForestRegressor()
RF_random_search = RandomizedSearchCV(estimator=RF_regressor, param_distributions=RF_random_grid,
                                      n_iter=100, cv=5, verbose=0, 
                                      random_state=2022, n_jobs = -1).fit(X_train, np.ravel(y_train))
RF_best_params = RF_random_search.best_params_

# Thu hẹp lưới tham số dựa trên các tham số tốt nhất được cung cấp bởi tìm kiếm ngẫu nhiên, sau đó đưa lưới vào tìm kiếm lưới
RF_param_grid = {'n_estimators': [RF_best_params['n_estimators']-100, RF_best_params['n_estimators'], 
                                  RF_best_params['n_estimators']+100],
               'max_features': ['sqrt', 'log2'],
               'max_depth': [RF_best_params['max_depth'] - 10, RF_best_params['max_depth'], 
                             RF_best_params['max_depth']+10],
               'min_samples_split': [5, 10],
               'min_samples_leaf': [1, 2],
               'bootstrap': [True, False]}

# Tạo ra một mô hình khác để fit với grid search
RF_regressor_2 = RandomForestRegressor()
RF_grid_search = GridSearchCV(estimator=RF_regressor_2, param_grid=RF_param_grid, 
                              cv=3, n_jobs=-1, verbose=0).fit(X_train, np.ravel(y_train))


# In kết quả tham số
print(RF_grid_search.best_params_)


# Sử dụng mô hình với bộ tham số tìm được
RF = RF_grid_search.best_estimator_

# Dự đoán trên tập kiểm thử
RF_predictions = RF.predict(X_test)


# Kết thúc huấn luyện
training_done = True
thread.join()  # Đợi thread kết thúc

# Hiển thị thông báo khi mô hình đã hoàn thành
print("\nHuấn luyện mô hình hoàn tất!")

# Đánh giá mô hình bằng cách in ra các chỉ số
mae = mean_absolute_error(y_test, RF_predictions)
mse = mean_squared_error(y_test, RF_predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, RF_predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Lưu mô hình dự đoán
joblib.dump(RF,"F:\\Data Analyst\\Ha Noi Real Estate Visualization\\model\\RF_model.pkl")
