import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('output.csv')

# Chuyển cột dateCreated sang định dạng datetime
data['dateCreated'] = pd.to_datetime(data['dateCreated'], errors='coerce')

# Lựa chọn các cột quan trọng
selected_data = data[['dateCreated', 'ratingValue', 'ratingCount', 'genres']]

# Loại bỏ các dòng có giá trị NaN
selected_data = selected_data.dropna()

# Tạo hai hình vẽ riêng biệt cho 'ratingValue' và 'ratingCount'
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(selected_data['dateCreated'], selected_data['ratingValue'], label='Rating Value', color='blue')
plt.xlabel('Ngày tạo')
plt.ylabel('Giá trị')
plt.title('Phân tích Time Series cho Rating Value')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(selected_data['dateCreated'], selected_data['ratingCount'], label='Rating Count', color='red')
plt.xlabel('Ngày tạo')
plt.ylabel('Giá trị')
plt.title('Phân tích Time Series cho Rating Count')
plt.legend()

plt.tight_layout()
plt.show()
