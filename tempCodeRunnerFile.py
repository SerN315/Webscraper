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