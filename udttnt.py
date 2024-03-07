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
import seaborn as sns
import matplotlib.pyplot as plt

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

# Create a correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.show()



import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

# Get the coefficients of the model
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.bar(coefficients["Feature"], coefficients["Coefficient"])
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the absolute values of the coefficients
abs_coefficients = abs(model.coef_)

# Sort the coefficients in descending order
sorted_indices = abs_coefficients.argsort()[::-1]

# Select the top 3 most important features
top_3_indices = sorted_indices[:3]
top_3_features = X.columns[top_3_indices]

# Plot bar plots for the top 3 features with the target variable
for feature in top_3_features:
    plt.figure(figsize=(8, 6))
    grouped_data = df.groupby(feature)["quality"].mean()
    if len(grouped_data) > 10:
        grouped_data = grouped_data.sample(n=10)  # Limit to 10 random samples if more than 10 unique values
    grouped_data.plot(kind="bar")
    plt.xlabel(feature)
    plt.ylabel("Average Quality")
    plt.title(f"Bar Plot: Average Quality by {feature}")
    plt.show()