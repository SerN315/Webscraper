from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.options import Options
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from csv import DictWriter
import time
import os
from selenium.common.exceptions import TimeoutException
def scrape():
    options = Options()
    ua = UserAgent()
    user_agent = ua.random
    options.add_argument(f'user-agent={user_agent}')
    driver = webdriver.Chrome(options=options)
    try:
        driver.get("https://www.lazada.vn/")
        search_element = driver.find_element(By.XPATH, "//input[@class='search-box__input--O34g']")
        search_element.send_keys("smartphone") 
        search_button = driver.find_element(By.XPATH, "//button[@class='search-box__button--1oH7']")
        search_button.click()
        current_page_url = driver.current_url
        products_elements = driver.find_elements(By.CSS_SELECTOR, ".Bm3ON")
        data = []

        for i in range(min(len(products_elements),40)):
            products_element = products_elements[i]
            title_link_element = products_element.find_element(By.CSS_SELECTOR, ".Ms6aG .qmXQo .ICdUp ._95X4G a")
            link = title_link_element.get_attribute("href")
            driver.get(link)
            time.sleep(2)
            try:
                title_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//h1[@class='pdp-mod-product-badge-title']")))
            except TimeoutException:
                print("Timeout occurred while waiting for the title element. Skipping iteration.")
                continue
            try:
                price_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@class='pdp-product-price']//span[@class='pdp-price pdp-price_type_normal pdp-price_color_orange pdp-price_size_xl']"))
            )
            except TimeoutException:
                print("Timeout occurred while waiting for the price element. Skipping iteration.")
                continue
            body = driver.find_element(By.TAG_NAME, 'body')
            body.send_keys(Keys.PAGE_DOWN)
            body.send_keys(Keys.PAGE_DOWN)
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(10)
            try:
                summary_element = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".mod-rating .content .left .summary"))
                )
            except TimeoutException:
                print("Timeout occurred while waiting for the summary element. Skipping iteration.")
                continue

            rating_score = summary_element.find_element(By.CLASS_NAME, "score")
            rating_score_average = rating_score.find_element(By.CLASS_NAME, "score-average")
            star_score_average = summary_element.find_element(By.CSS_SELECTOR, ".average .container-star")
            star_images = star_score_average.find_elements(By.TAG_NAME, "img")
            rating_count = summary_element.find_element(By.CSS_SELECTOR, ".count")
            time.sleep(5)
            comments = driver.find_elements(By.CSS_SELECTOR, ".mod-reviews .item")
            usernames = []
            ratings = []
            comment_contents = []

            for comment in comments:
                star_urls_per_comment = []
                star_sums_per_comment = []
                star_score_per = comment.find_element(By.CSS_SELECTOR, ".top .container-star")
                star_images_per = star_score_per.find_elements(By.CSS_SELECTOR, ".starCtn .star")
                star_urls_per_comment = [star_image.get_attribute("src") for star_image in star_images_per]

                # Group the star URLs into chunks of 5
                grouped_star_urls = [star_urls_per_comment[i:i + 5] for i in range(0, len(star_urls_per_comment), 5)]

                # Calculate the sums for each group of 5 star URLs
                for group in grouped_star_urls:
                    star_sum_per_group = sum(1 if url == "https://laz-img-cdn.alicdn.com/tfs/TB19ZvEgfDH8KJjy1XcXXcpdXXa-64-64.png" else 0 for url in group)
                    star_sums_per_comment.append(star_sum_per_group)

                # Calculate the overall rating for each comment
                overall_rating_per_comment = sum(star_sums_per_comment) / len(star_sums_per_comment)

                # Round the overall rating to the nearest integer
                overall_rating_per_comment_int = round(overall_rating_per_comment)

                username = comment.find_element(By.CSS_SELECTOR, ".middle span:first-child").text
                usernames.append(username)
                ratings.append(overall_rating_per_comment_int)
                comment_content = comment.find_element(By.CSS_SELECTOR, ".item-content .content")
                comment_text = comment_content.text
                comment_contents.append(comment_text)

            # Ensure all lists have the same number of elements
            min_len = min(len(usernames), len(ratings), len(comment_contents))
            usernames = usernames[:min_len]
            ratings = ratings[:min_len]
            comment_contents = comment_contents[:min_len]

            # Append the data for each product to the overall data list
            row_data = {
                "link": link,
                "price": price_element.text,
                "TenSP": title_element.text,
                "DG": rating_score_average.text,
                "SoDG": rating_count.text,
                "usernames": usernames,
                "ratings": ratings,
                "comments": comment_contents
            }
            data.append(row_data)

            driver.back()
            products_elements = driver.find_elements(By.CSS_SELECTOR, ".Bm3ON")

        write_headers = not os.path.exists('lazada.csv')

        with open('lazada.csv', 'w', newline='', encoding='utf-8') as file:
            writer = DictWriter(file, fieldnames=["TenSP", "price", "link", "DG", "SoDG", "usernames", "ratings", "comments"])
            if write_headers:
                writer.writeheader()
            writer.writerows(data)
    finally:
        driver.quit()

# Call the scrape function to perform scraping
scrape()



import os
import sqlite3
from csv import DictReader

def create_database():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT,
            price TEXT,
            TenSP TEXT,
            DG TEXT,
            SoDG TEXT,
            source TEXT DEFAULT 'lazada'
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            username TEXT,
            rating INTEGER,
            comment TEXT,
            FOREIGN KEY (product_id) REFERENCES products (id)
        )
    ''')

    conn.commit()
    conn.close()

def insert_data_into_database(row_data):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    # Insert product data with the 'source' column
    cursor.execute('''
        INSERT INTO products (link, price, TenSP, DG, SoDG, source) VALUES (?, ?, ?, ?, ?, ?)
    ''', (row_data['link'], row_data['price'], row_data['TenSP'], row_data['DG'], row_data['SoDG'], 'lazada'))

    product_id = cursor.lastrowid  # Get the ID of the last inserted product

    # Insert comments data without the 'source' column
    usernames = row_data.get('usernames', [])
    ratings = row_data.get('ratings', [])
    comments = row_data.get('comments', [])

    for i in range(len(usernames)):
        username = usernames[i].strip("[]'") if i < len(usernames) else ''
        rating = ratings[i].strip("[]'") if i < len(ratings) else ''
        comment = comments[i].strip("[]'") if i < len(comments) else ''

        cursor.execute('''
            INSERT INTO comments (product_id, username, rating, comment) VALUES (?, ?, ?, ?)
        ''', (product_id, username, rating, comment))

    conn.commit()
    conn.close()

def format_data_and_insert():
    write_headers = not os.path.exists('lazada.csv')

    with open('lazada.csv', 'r', encoding='utf-8') as file:
        csv_reader = DictReader(file)
        for row in csv_reader:
            usernames = row.get("TenHienthi", "").split(", ")  # Assume usernames are comma-separated
            ratings = row.get("comment_DG", "").split(", ")  # Assume ratings are comma-separated
            comment_contents = row.get("comment", "").split(", ")  # Assume comments are comma-separated

            # Append the data for each product to the overall data list
            row_data = {
                "link": row.get("link", ""),
                "price": row.get("price", ""),
                "TenSP": row.get("TenSP", ""),
                "DG": row.get("DG", ""),
                "SoDG": row.get("SoDG", ""),
                "usernames": usernames,
                "ratings": ratings,
                "comments": comment_contents
            }

            # Insert data into SQLite database
            insert_data_into_database(row_data)

# Create the database and tables
create_database()

# Format data from CSV and insert into SQLite database
format_data_and_insert()


#facecollector
import cv2
import os

data_dir = "data"
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
id = input("Chọn ID người dùng: ")
emotion_label = input(" Nhãn cảm xúc (Angry, Disgust, Fear, Happy, Sad,Surprise, Neutral): ")
count = 0

while True:
    ret, frame = video.read()
    # Chuyển đổi ảnh về màu xám để giảm độ phức tạp của ảnh
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Sử dụng hàm detectMultiScale để nhận diện khuôn mặt
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        # Lưu ảnh khuôn mặt với tên file có định dạng "User.ID.count.emotion.jpg"
        file_name = f"User.{id}.{count}.{emotion_label}.jpg"
        cv2.imwrite(os.path.join(data_dir, file_name), face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (119, 221, 119), 1) 
        
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    
    if count > 50:
        break
    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
print("Đã thu thập dữ liệu và gán nhãn cảm xúc thành công!")




import face_recognition
import numpy as np
import os
import matplotlib.pyplot as plt

path = "data"

def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    emotion_labels = []
    ids = []

    for imagePaths in imagePath:
        faceImage = face_recognition.load_image_file(imagePaths)
        faceLocations = face_recognition.face_locations(faceImage)

        if len(faceLocations) > 0:
            faceNP = face_recognition.face_encodings(faceImage, faceLocations)[0]
            emotion_label = os.path.split(imagePaths)[-1].split(".")[3]
            Id = int(os.path.split(imagePaths)[-1].split(".")[1])

            faces.append(faceNP)
            emotion_labels.append(emotion_label)
            ids.append(Id)

    return ids, faces, emotion_labels

IDs, facedata, emotion_labels = getImageID(path)
np.savez("Trainer.npz", facedata=facedata, IDs=IDs, emotion_labels=emotion_labels)
plt.close('all')
print("Đã xử lý dữ liệu thành công!")


import cv2
import face_recognition
import numpy as np
import keras
from tensorflow.keras.models import load_model

# Load the pre-trained face recognition model
known_faces = np.load("Trainer.npz")
facedata = known_faces["facedata"]
IDs = known_faces["IDs"]

# Load the pre-trained emotion detection model
emotion_model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate to 60 fps
name_list = ["", "Nguyen"]

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        # Extract the face encoding from the current frame
        face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]

        # Perform face recognition using the known face encodings
        face_distances = face_recognition.face_distance(facedata, face_encoding)
        min_distance_index = np.argmin(face_distances)
        min_distance = face_distances[min_distance_index]

        if min_distance < 0.6:
            name = name_list[IDs[min_distance_index]]
        else:
            name = "Khong biet luon"

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (119, 221, 119), 1)
        cv2.putText(frame, name, (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (119, 221, 119), 2)

        # Extract the face ROI for emotion detection
        face_roi = frame[top:bottom, left:right]
        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi_gray = cv2.resize(face_roi_gray, (64, 64))
        face_roi_gray = face_roi_gray.astype("float") / 255.0
        face_roi_gray = np.expand_dims(face_roi_gray, 0)
        face_roi_gray = np.expand_dims(face_roi_gray, -1)

        # Perform emotion detection
        emotion_preds = emotion_model.predict(face_roi_gray)[0]
        emotion_label = emotion_labels[np.argmax(emotion_preds)]

        # Display the predicted emotion
        cv2.putText(frame, emotion_label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (119, 221, 119), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

