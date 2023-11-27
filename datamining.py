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
    movie_title = 'Invincible Season 2'
    recommendations = get_movie_recommendations(data, movie_title, cosine_sim)
    print(f"Các đề xuất cho '{movie_title}': {recommendations}")

if __name__ == "__main__":
    main()


# Đoạn code trên thực hiện một hệ thống đề xuất phim dựa trên nội dung, sử dụng các phương pháp xử lý văn bản và tính toán độ tương đồng giữa các bộ phim.

# 1. **Tải và Chuẩn bị Dữ liệu:**
#    - Dữ liệu được tải từ tệp CSV ('output.csv') chứa thông tin về các bộ phim.
#    - Các cột 'ratingValue' và 'ratingCount' được làm sạch để chuyển đổi về dạng số.

# 2. **Kết hợp Các Đặc Trưng Liên Quan:**
#    - Các thông tin có liên quan của mỗi bộ phim được kết hợp lại thành một cột mới ('combined_features'). Các thông tin này bao gồm tiêu đề phim, mô tả, đánh giá, số lượt đánh giá và thể loại của phim.

# 3. **Vector hóa TF-IDF cho Văn Bản:**
#    - Sử dụng `TfidfVectorizer` để chuyển đổi các văn bản thành ma trận dữ liệu TF-IDF.
#    - TF-IDF đo lường tần suất xuất hiện của từng từ trong một văn bản so với toàn bộ tập dữ liệu và giúp đánh giá mức độ quan trọng của từng từ đối với mỗi mẫu dữ liệu.

# 4. **Chuẩn hóa Các Đặc Trưng Số:**
#    - Điều chỉnh các đặc trưng số như 'ratingValue' và 'ratingCount' thành khoảng giá trị chuẩn hóa.

# 5. **Tính Ma Trận Tương Đồng:**
#    - Sử dụng `cosine_similarity` để tính toán ma trận độ tương đồng cosine giữa các bộ phim dựa trên các đặc trưng kết hợp và ma trận TF-IDF.

# 6. **Hàm Đề Xuất Phim:**
#    - Hàm `get_movie_recommendations` nhận đầu vào là tựa đề phim và trả về danh sách các bộ phim được đề xuất dựa trên độ tương đồng cosine của nó với các bộ phim khác.

#    Giải thích các thuật toán:
#    - **TF-IDF (Term Frequency-Inverse Document Frequency):** Đánh giá tần suất xuất hiện của từ trong một văn bản so với toàn bộ tập dữ liệu. Tính toán giá trị quan trọng của từng từ trong một văn bản.
#    - **Cosine Similarity:** Đo lường độ giống nhau giữa hai véc-tơ dựa trên góc giữa chúng trong không gian đa chiều. Trong trường hợp này, nó đo lường sự tương đồng giữa các bộ phim dựa trên các đặc trưng được kết hợp và văn bản của chúng.

# Mục tiêu của đoạn code này là cung cấp gợi ý bộ phim dựa trên sự tương đồng về nội dung giữa các bộ phim trong tập dữ liệu.




# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA

# # Đọc dữ liệu từ file CSV và chuyển cột dateCreated sang định dạng datetime
# data = pd.read_csv('output.csv')
# data['dateCreated'] = pd.to_datetime(data['dateCreated'], errors='coerce')
# data['ratingValue'] = pd.to_numeric(data['ratingValue'], errors='coerce')
# data['ratingCount'] = pd.to_numeric(data['ratingCount'], errors='coerce')

# # Lựa chọn các cột quan trọng
# selected_data = data[['dateCreated', 'ratingCount']]

# # Loại bỏ các dòng có giá trị NaN
# selected_data = selected_data.dropna()

# # Chia dữ liệu thành train và test
# train_data = selected_data.iloc[:-300]  # Sử dụng 10 dòng cuối cùng làm test

# # Xây dựng mô hình ARIMA
# model = ARIMA(train_data['ratingCount'], order=(5,1,0))  # Chọn order phù hợp
# model_fit = model.fit()

# # Dự đoán
# forecast = model_fit.forecast(steps=10)  # Dự đoán cho 10 bước tiếp theo

# # Tạo chuỗi thời gian mới cho dự đoán
# last_date = train_data['dateCreated'].iloc[-1]
# forecast_dates = pd.date_range(start=last_date, periods=11, freq='M')[1:]

# # Đánh giá mô hình
# model_fit.plot_diagnostics(figsize=(15, 12))
# plt.show()

# # So sánh dự đoán với dữ liệu thực tế
# test_data = selected_data.iloc[-300:]
# plt.figure(figsize=(12, 6))
# plt.plot(test_data['dateCreated'], test_data['ratingCount'], label='Actual', color='blue')
# plt.plot(forecast_dates, forecast, label='Forecast', color='green')
# plt.xlabel('Ngày tạo')
# plt.ylabel('Rating Count')
# plt.title('So sánh dự đoán với dữ liệu thực tế')
# plt.legend()
# plt.show()

# # Tối ưu hóa mô hình
# best_aic = float("inf")
# best_order = None

# for p in range(3):
#     for d in range(3):
#         for q in range(3):
#             try:
#                 model = ARIMA(train_data['ratingCount'], order=(p,d,q))
#                 model_fit = model.fit()
#                 if model_fit.aic < best_aic:
#                     best_aic = model_fit.aic
#                     best_order = (p, d, q)
#             except:
#                 continue

# print(f"Best AIC: {best_aic}, Best Order: {best_order}")

# # Xây dựng lại mô hình với tham số tốt nhất
# best_model = ARIMA(train_data['ratingCount'], order=best_order)
# best_model_fit = best_model.fit()

# # Dự đoán với mô hình tối ưu hóa
# best_forecast = best_model_fit.forecast(steps=10)

# # Biểu đồ dự đoán mới
# plt.figure(figsize=(12, 6))
# plt.plot(selected_data['dateCreated'], selected_data['ratingCount'], label='Actual', color='blue')
# plt.plot(train_data['dateCreated'], best_model_fit.fittedvalues, label='Fitted', color='red')
# plt.plot(forecast_dates, best_forecast, label='Forecast', color='green')
# plt.xlabel('Ngày tạo')
# plt.ylabel('Rating Count')
# plt.title('Dự đoán chuỗi thời gian cho Rating Count (Mô hình tối ưu hóa)')
# plt.legend()
# plt.show()
