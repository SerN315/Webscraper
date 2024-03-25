from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.options import Options
from selenium import webdriver
from csv import DictWriter
import time
import os

def scrape():
    options = Options()
    options.headless = False  

    driver = webdriver.Edge()

    try:
        # Truy cập vào trang web
        driver.get("https://phimmoiyyy.net/")
        time.sleep(2)

        # Tìm phần tử link "2023" và click vào nó
        link_element = driver.find_element(By.LINK_TEXT, "2023")
        link_element.click()
        time.sleep(2)

        while True:
            # Chờ cho đến khi URL chứa "/nam-phat-hanh/2023"
            WebDriverWait(driver, 5).until(EC.url_contains("/nam-phat-hanh/2023"))
            current_page_url = driver.current_url  # Lưu trữ URL trang hiện tại
            # Lấy danh sách các phần tử article trên trang
            article_elements = driver.find_elements(By.CSS_SELECTOR, "article.item")

            data = []

            # Lặp qua các phần tử article
            for i in range(len(article_elements)):
                article_element = article_elements[i]

                # Lấy phần tử link tiêu đề và lấy thuộc tính href
                title_link_element = article_element.find_element(By.CSS_SELECTOR, "h3 a")
                link = title_link_element.get_attribute("href")

                # Truy cập vào trang con
                driver.get(link)

                # Lấy các thông tin cần thiết từ trang con
                info_element = driver.find_element(By.CSS_SELECTOR, "#info p")
                title_element = driver.find_element(By.CSS_SELECTOR, "h1")
                date_element = driver.find_element(By.CSS_SELECTOR, ".date")
                rating_value_element = driver.find_element(By.CSS_SELECTOR, ".starstruck-rating span.dt_rating_vgs")
                rating_count_element = driver.find_element(By.CSS_SELECTOR, ".starstruck-rating span.rating-count")
                
                genres_elements = driver.find_elements(By.CSS_SELECTOR, ".sgeneros a")

                # Lấy nội dung của các phần tử khác
                info = info_element.text
                title = title_element.text
                date_created = date_element.text
                rating_value = rating_value_element.text
                rating_count = rating_count_element.text

                # Lấy các thể loại từ các phần tử a trong phần tử có class "sgeneros"
                genres = [genre_element.text for genre_element in genres_elements]

                # Tạo đối tượng dữ liệu cho hàng dữ liệu hiện tại
                row_data = {
                    "info": info,
                    "title": title,
                    "dateCreated": date_created,
                    "ratingValue": rating_value,
                    "ratingCount": rating_count,
                    "genres": genres
                }

                # Thêm hàng dữ liệu vào mảng data
                data.append(row_data)

                # Quay lại trang danh sách
                driver.get(current_page_url)
                article_elements = driver.find_elements(By.CSS_SELECTOR, "article.item")

            # Kiểm tra xem tập tin đã tồn tại chưa để xác định việc ghi headers hay không
            write_headers = not os.path.exists("output.csv")

            # Ghi dữ liệu vào file CSV
            with open("output.csv", "a", newline="", encoding="utf-8") as csv_file:
                fieldnames = [
                    "title", "info", "dateCreated", "ratingValue", "ratingCount", "genres"
                ]
                writer = DictWriter(csv_file, fieldnames=fieldnames)
                write_headers = True

                if write_headers:
                    writer.writeheader()  # Ghi headers chỉ khi tập tin mới

                writer.writerows(data)

            # Kiểm tra xem có trang tiếp theo không
            next_page_links = driver.find_elements(By.CSS_SELECTOR, '.pagination a #nextpagination')
            next_page_link = next((link for link in next_page_links if 'inactive' not in link.get_attribute('class')), None)

            if next_page_link:
                next_page_number = int(next_page_link.text) if next_page_link.text.isdigit() else 0

            if not next_page_link or current_page_url.endswith(f"page/{next_page_number}"):
                break  # Thoát vòng lặp nếu không có trang tiếp theo hoặc đang ở trang hiện tại

            # Click vào liên kết trang tiếp theo bằng JavaScript
            driver.execute_script("arguments[0].click();", next_page_link)
            time.sleep(2)  # Thêm độ trễ nhỏ để trang mới tải

            # Chờ cho trang mới tải hoàn toàn
            WebDriverWait(driver, 10).until(EC.url_changes(current_page_url))
            current_page_url = driver.current_url  # Cập nhật URL trang hiện tại cho vòng lặp tiếp theo

    finally:
        # Đóng trình duyệt
        driver.quit()

# Gọi hàm scrape
scrape()


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Tải dữ liệu từ tệp CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Dọn dẹp các cột số bằng cách chuyển chúng thành kiểu số
def clean_numeric_columns(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    return df

# Điền giá trị thiếu trong cột 'info' và 'genres' bằng chuỗi trống
def fill_missing_values(df):
    df['info'] = df['info'].fillna('')
    df['genres'] = df['genres'].fillna('')
    return df

# Kết hợp các đặc trưng khác nhau thành một đặc trưng duy nhất
def combine_features(df):
    df['combined_features'] = df['title'] + ' ' + df['info'] + ' ' + df['ratingValue'].astype(str) + ' ' + df['ratingCount'].astype(str) + ' ' + df['genres']
    return df

# Tiền xử lý dữ liệu (dọn dẹp, xử lý giá trị thiếu, kết hợp đặc trưng)
def preprocess_data(data):
    data = clean_numeric_columns(data, ['ratingValue', 'ratingCount'])
    data = fill_missing_values(data)
    data = combine_features(data)
    return data

# Vector hóa đặc trưng văn bản sử dụng TF-IDF
def vectorize_text_features(features):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(features)
    return tfidf_matrix

# Chuẩn hóa các đặc trưng số bằng MinMaxScaler
def normalize_numerical_features(data):
    scaler = MinMaxScaler()
    numerical_features = data[['ratingValue', 'ratingCount']].astype(float)
    data[['ratingValue', 'ratingCount']] = scaler.fit_transform(numerical_features)
    return data

# Tính độ tương đồng cosine giữa các mục dựa trên đặc trưng
def calculate_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Lấy các đề xuất phim dựa trên điểm tương đồng
def get_movie_recommendations(data, movie_title, cosine_sim, top_n=10):
    filtered_data = data[data['title'] == movie_title]
    if filtered_data.empty:
        print(f"Không tìm thấy phim '{movie_title}' trong dữ liệu.")
        return []

    idx = filtered_data.index[0] if len(filtered_data) > 0 else None

    if idx is None:
        print(f"Không có phim tương tự cho '{movie_title}'.")
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:top_n+1]]  # Cắt để lấy top_n đề xuất
    return data['title'].iloc[movie_indices].tolist()

# Hàm chính điều hành toàn bộ quá trình
def main():
    # Tải dữ liệu
    file_path = 'output.csv'
    data = load_data(file_path)

    # Tiền xử lý dữ liệu
    data = preprocess_data(data)

    # Vector hóa đặc trưng văn bản
    tfidf_matrix = vectorize_text_features(data['combined_features'])

    # Chuẩn hóa các đặc trưng số
    data = normalize_numerical_features(data)

    # Tính độ tương đồng cosine
    cosine_sim = calculate_similarity(tfidf_matrix)

    # Lấy các đề xuất phim
    movie_title = 'Vệ Binh Dải Ngân Hà 3'
    recommendations = get_movie_recommendations(data, movie_title, cosine_sim)
    print(f"Các đề xuất cho '{movie_title}': {recommendations}")

if __name__ == "__main__":
    main()



























import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import numpy as np
import fasttext.util
from langdetect import detect


# Tải dữ liệu từ tệp CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Dọn dẹp các cột số bằng cách chuyển chúng thành kiểu số
def clean_numeric_columns(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    return df

# Điền giá trị thiếu trong cột 'info' và 'genres' bằng chuỗi trống
def fill_missing_values(df):
    df['info'] = df['info'].fillna('')
    df['genres'] = df['genres'].fillna('')
    return df

# Kết hợp các đặc trưng khác nhau thành một đặc trưng duy nhất
def combine_features(df):
    df['combined_features'] = df['title'] + ' ' + df['info'] + ' ' + df['ratingValue'].astype(str) + ' ' + df['ratingCount'].astype(str) + ' ' + df['genres']
    return df

# Tiền xử lý dữ liệu (dọn dẹp, xử lý giá trị thiếu, kết hợp đặc trưng)
def preprocess_data(data):
    data = clean_numeric_columns(data, ['ratingValue', 'ratingCount'])
    data = fill_missing_values(data)
    data = combine_features(data)
    return data

# Vector hóa đặc trưng văn bản sử dụng TF-IDF
def vectorize_text_features(features):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(features)
    return tfidf_matrix

# Chuẩn hóa các đặc trưng số bằng MinMaxScaler
def normalize_numerical_features(data):
    scaler = MinMaxScaler()
    numerical_features = data[['ratingValue', 'ratingCount']].astype(float)
    data[['ratingValue', 'ratingCount']] = scaler.fit_transform(numerical_features)
    return data

# Train RNN model
def train_rnn_model(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['combined_features'])
    sequences = tokenizer.texts_to_sequences(data['combined_features'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)  # Define max_len
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(units=64))
    model.add(Dense(units=1, activation='sigmoid'))  # Adjust units and activation as needed
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Adjust loss and metrics
    model.fit(padded_sequences, data['ratingValue'], epochs=10, batch_size=32)  # Adjust epochs and batch_size
    return model

# Train LDA model
def train_lda_model(data,tfidf_matrix):
    # Preprocess text for LDA (tokenization, removing stopwords, stemming, etc.)
    # Then train LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)
    return lda

# Combine RNN and LDA representations
def combine_representations(rnn_model, lda_model, tfidf_matrix,padded_sequences):
    rnn_representations = rnn_model.predict(padded_sequences)  # Assuming padded_sequences are defined
    lda_representations = lda_model.transform(tfidf_matrix)
    combined_representations = np.concatenate((rnn_representations, lda_representations), axis=1)
    return combined_representations

# Calculate similarity between movies
def calculate_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Calculate similarity between user input and movies
def calculate_similarity_with_user_input(user_embedding, movie_embeddings):
    return cosine_similarity(user_embedding, movie_embeddings)

# Get top similar movies
def get_top_similar_movies(data, similarity_matrix, n=10):
    top_indices = np.argsort(similarity_matrix)[-n-1:-1][::-1]  # Exclude self-similarity
    return data.iloc[top_indices]

# Main function
def main():
    # Load data
    file_path = 'output.csv'
    data = load_data(file_path)
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Vectorize text features
    tfidf_matrix = vectorize_text_features(data['combined_features'])
    
    # Normalize numerical features
    data = normalize_numerical_features(data)
    
    # Train RNN model
    rnn_model = train_rnn_model(data)
    
    # Train LDA model
    lda_model = train_lda_model(data)
    
    # Combine representations
    combined_representations = combine_representations(rnn_model, lda_model, tfidf_matrix)
    
    # Calculate similarity
    cosine_sim = calculate_similarity(combined_representations)
    
    # Get user input (description can be in English or Vietnamese)
    user_description = input("Enter a movie description: ")
    
    # Identify language of user input
    lang = identify_language(user_description)
    
    # Preprocess user input
    preprocessed_user_input = preprocess_text(user_description, lang)
    
    # Generate embeddings for user input
    user_embedding = rnn_model.predict(pad_sequences(tokenizer.texts_to_sequences([preprocessed_user_input]), maxlen=max_len))
    
    # Calculate similarity between user input and movies
    user_movie_similarity = calculate_similarity_with_user_input(user_embedding, combined_representations)
    
    # Get top similar movies
    top_similar_movies = get_top_similar_movies(data, user_movie_similarity)
    
    print("Top similar movies:")
    print(top_similar_movies)

# Parameters
num_words = 10000  # Number of words in tokenizer
embedding_dim = 100  # Embedding dimension
max_len = 100  # Maximum sequence length for padding
n_topics = 10  # Number of topics for LDA

# Call main function
if __name__ == "__main__":
    main()
