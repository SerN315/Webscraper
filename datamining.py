import csv
from csv import DictWriter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up the Chrome driver
driver = webdriver.Edge()

# Xác định hàm scraping
def scrape():
    try:
        # Truy cập vào trang web
        driver.get("https://phimmoiyyy.net/")

        # Tìm phần tử link "2023" và click vào nó
        link_element = driver.find_element(By.LINK_TEXT, "2023")
        link_element.click()

        # Chờ cho đến khi URL chứa "/nam-phat-hanh/2023"
        WebDriverWait(driver, 5).until(EC.url_contains("/nam-phat-hanh/2023"))

        # Lấy danh sách các phần tử article trên trang
        article_elements = driver.find_elements(By.CSS_SELECTOR, "article.item")

        data = []

        # Lặp qua 3 phần tử article đầu tiên
        for i in range(len(article_elements)):
            article_element = article_elements[i]

            # Lấy phần tử link tiêu đề và lấy thuộc tính href
            title_link_element = article_element.find_element(By.CSS_SELECTOR, "h3 a")
            link = title_link_element.get_attribute("href")

            # Truy cập vào trang con
            driver.get(link)

            # Lấy các thông tin cần thiết từ trang con
            img_element = driver.find_element(By.CSS_SELECTOR, ".poster img")
            title_element = driver.find_element(By.CSS_SELECTOR, "h1")
            span_content_element = driver.find_element(By.CSS_SELECTOR, ".extra .valor")
            date_element = driver.find_element(By.CSS_SELECTOR, ".date")
            rating_value_element = driver.find_element(By.CSS_SELECTOR, ".starstruck-rating span.dt_rating_vgs")
            rating_count_element = driver.find_element(By.CSS_SELECTOR, ".starstruck-rating span.rating-count")
            genres_elements = driver.find_elements(By.CSS_SELECTOR, ".sgeneros a")

            # Lấy giá trị của thuộc tính src từ phần tử img
            img_url = img_element.get_attribute("src")

            # Lấy nội dung của các phần tử khác
            title = title_element.text
            span_content = span_content_element.text
            date_created = date_element.text
            rating_value = rating_value_element.text
            rating_count = rating_count_element.text

            # Lấy các thể loại từ các phần tử a trong phần tử có class "sgeneros"
            genres = [genre_element.text for genre_element in genres_elements]

            # Tạo đối tượng dữ liệu cho hàng dữ liệu hiện tại
            row_data = {
                "imgUrl": img_url,
                "title": title,
                "link": link,
                "spanContent": span_content,
                "dateCreated": date_created,
                "ratingValue": rating_value,
                "ratingCount": rating_count,
                "genres": genres
            }

            # Thêm hàng dữ liệu vào mảng data
            data.append(row_data)

            # Quay lại trang danh sách
            driver.get("https://phimmoiyyy.net/")
            link_element = driver.find_element(By.LINK_TEXT, "2023")
            link_element.click()
            WebDriverWait(driver, 5).until(EC.url_contains("/nam-phat-hanh/2023"))
            article_elements = driver.find_elements(By.CSS_SELECTOR, "article.item")

        # Ghi dữ liệu vào file CSV
        with open("output.csv", "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "imgUrl", "title", "link", "spanContent",
                "dateCreated", "ratingValue", "ratingCount", "genres"
            ]
            writer = DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    finally:
        # Close the browser
        driver.quit()

# Call the scrape function
scrape()


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