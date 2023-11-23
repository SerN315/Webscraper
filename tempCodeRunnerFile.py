import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz

# Load the scraped data from the CSV file
data = []
with open("output.csv", "r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        data.append(row)

# Prepare the feature matrix X and the target variable y
X = []
y = []
for row in data:
    rating_value = float(row["ratingValue"])
    genres = row["genres"]
    X.append([rating_value])
    y.append(genres)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the decision tree classifier and fit it to the training data
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Use the trained classifier to make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the decision tree
dot_data = export_graphviz(clf, out_file=None, feature_names=["ratingValue"], class_names=clf.classes_, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render(filename="decision_tree", format="png", engine="dot", directory="C:\coding\Html\Webscraper\decision_tree")

# Predict the genres based on a user's rating
user_rating = 4.5  # Example user rating
predicted_genres = clf.predict([[user_rating]])
print("Predicted genres:", predicted_genres)