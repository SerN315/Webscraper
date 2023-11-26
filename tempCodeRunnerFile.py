import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Đọc dữ liệu từ file CSV và chuyển cột dateCreated sang định dạng datetime
data = pd.read_csv('output.csv')
data['dateCreated'] = pd.to_datetime(data['dateCreated'], errors='coerce')
data['ratingValue'] = pd.to_numeric(data['ratingValue'], errors='coerce')
data['ratingCount'] = pd.to_numeric(data['ratingCount'], errors='coerce')

# Lựa chọn các cột quan trọng
selected_data = data[['dateCreated', 'ratingCount']]

# Loại bỏ các dòng có giá trị NaN
selected_data = selected_data.dropna()

# Chia dữ liệu thành train và test
train_data = selected_data.iloc[:-300]  # Sử dụng 10 dòng cuối cùng làm test

# Xây dựng mô hình ARIMA
model = ARIMA(train_data['ratingCount'], order=(5,1,0))  # Chọn order phù hợp
model_fit = model.fit()

# Dự đoán
forecast = model_fit.forecast(steps=10)  # Dự đoán cho 10 bước tiếp theo

# Tạo chuỗi thời gian mới cho dự đoán
last_date = train_data['dateCreated'].iloc[-1]
forecast_dates = pd.date_range(start=last_date, periods=11, freq='M')[1:]

# Đánh giá mô hình
model_fit.plot_diagnostics(figsize=(15, 12))
plt.show()

# So sánh dự đoán với dữ liệu thực tế
test_data = selected_data.iloc[-300:]
plt.figure(figsize=(12, 6))
plt.plot(test_data['dateCreated'], test_data['ratingCount'], label='Actual', color='blue')
plt.plot(forecast_dates, forecast, label='Forecast', color='green')
plt.xlabel('Ngày tạo')
plt.ylabel('Rating Count')
plt.title('So sánh dự đoán với dữ liệu thực tế')
plt.legend()
plt.show()

# Tối ưu hóa mô hình
best_aic = float("inf")
best_order = None

for p in range(3):
    for d in range(3):
        for q in range(3):
            try:
                model = ARIMA(train_data['ratingCount'], order=(p,d,q))
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = (p, d, q)
            except:
                continue

print(f"Best AIC: {best_aic}, Best Order: {best_order}")

# Xây dựng lại mô hình với tham số tốt nhất
best_model = ARIMA(train_data['ratingCount'], order=best_order)
best_model_fit = best_model.fit()

# Dự đoán với mô hình tối ưu hóa
best_forecast = best_model_fit.forecast(steps=10)

# Biểu đồ dự đoán mới
plt.figure(figsize=(12, 6))
plt.plot(selected_data['dateCreated'], selected_data['ratingCount'], label='Actual', color='blue')
plt.plot(train_data['dateCreated'], best_model_fit.fittedvalues, label='Fitted', color='red')
plt.plot(forecast_dates, best_forecast, label='Forecast', color='green')
plt.xlabel('Ngày tạo')
plt.ylabel('Rating Count')
plt.title('Dự đoán chuỗi thời gian cho Rating Count (Mô hình tối ưu hóa)')
plt.legend()
plt.show()
