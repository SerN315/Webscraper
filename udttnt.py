import csv

file_path = "winequality-red.csv"  # Replace with the actual file path

with open(file_path, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        print("fixed acidity: Nồng độ axit tổng", row["fixed acidity"])
        print("volatile acidity: Tinh axit", row["volatile acidity"])
        print("citric acid: Nồng độ axit Citric", row["citric acid"])
        print("residual sugar: Nồng độ đường dư", row["residual sugar"])
        print("chlorides: Nồng độ clo", row["chlorides"])
        print("free sulfur dioxide: Nồng độ acid sulfurus tự do", row["free sulfur dioxide"])
        print("density: Mật độ (khối lượng/don tích)", row["density"])
        print("sulphates: Nồng độ sulphates", row["sulphates"])
        print("pH:", row["pH"])
        print("alcohol:Nồng độ cồn",row["alcohol"])
        print()


import pandas as pd

file_path = "winequality-red.csv"  # Replace with the actual file path

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Calculate the statistics for the "alcohol" and "quality" columns
alcohol_mean = df["alcohol"].mean()
alcohol_median = df["alcohol"].median()
alcohol_correlation = df["alcohol"].corr(df["quality"])

# Print the results
print("Alcohol Mean:", alcohol_mean)
print("Alcohol Median:", alcohol_median)
print("Alcohol-Quality Correlation:", alcohol_correlation)




import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

file_path = "winequality-red.csv"  # Replace with the actual file path

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Define the columns of interest
columns_of_interest = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "density", "sulphates", "pH", "alcohol", "quality"
]

# Subset the DataFrame with the columns of interest
subset_df = df[columns_of_interest]

# Split the data into features (X) and target variable (y)
X = subset_df.drop("quality", axis=1)
y = subset_df["quality"]

# Perform data normalization using Min-Max scaling
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_normalized, y)

# Print the coefficients of the model
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print(coefficients)







import pandas as pd
from sklearn.linear_model import LinearRegression

file_path = "winequality-red.csv"  # Thay bằng đường dẫn thực tế tới file

# Đọc file CSV thành DataFrame của pandas
df = pd.read_csv(file_path)

# Định nghĩa các cột quan tâm
columns_of_interest = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "density", "sulphates", "pH", "alcohol", "quality"
]

# Lấy các cột quan tâm từ DataFrame
subset_df = df[columns_of_interest]

# Split the data into features (X) and target variable (y)
X = subset_df.drop("quality", axis=1)
y = subset_df["quality"]

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Lấy giá trị tuyệt đối của các hệ số
abs_coefficients = abs(model.coef_)

# Sắp xếp theo giá trị tuyệt đối giảm dần
sorted_indices = abs_coefficients.argsort()[::-1]

# Chọn ra 3 thuộc tính quan trọng nhất
top_3_indices = sorted_indices[:3]
top_3_features = X.columns[top_3_indices]

# In ra 3 thuộc tính quan trọng nhất và tương quan tuyến tính với quality
for feature in top_3_features:
    correlation = df[feature].corr(df["quality"])
    print(f"{feature}: {correlation}")