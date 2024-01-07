import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu từ file CSV
data = pd.read_csv('output.csv')  # Thay 'your_data.csv' bằng tên file CSV chứa dữ liệu

# Tiền xử lý dữ liệu
# Tách các giá trị thể loại thành các cột riêng biệt
data['genres'] = data['genres'].apply(eval)  # Chuyển chuỗi thể loại thành list
genres_columns = pd.get_dummies(data['genres'].apply(pd.Series).stack()).sum(level=0)
data = pd.concat([data, genres_columns], axis=1)

# Lựa chọn các cột quan trọng cho việc huấn luyện mô hình
selected_columns = ['ratingValue', 'ratingCount'] + genres_columns.columns.tolist()
X = pd.concat([data[selected_columns], genres_columns], axis=1)
y = data['potential_success']  # Assumed label for potential success

# Chia dữ liệu thành train và test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Báo cáo classification
print(classification_report(y_test, y_pred))
