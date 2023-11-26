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
    options.headless = False  # Set to True if you want it to run in the background

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
            current_page_url = driver.current_url  # Store the current page UR
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

                # Lấy giá trị của thuộc tính src từ phần tử info
                

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

            # Check if the file already exists to determine whether to write headers or not
            write_headers = not os.path.exists("output.csv")

            # Ghi dữ liệu vào file CSV
            with open("output.csv", "a", newline="", encoding="utf-8") as csv_file:
               fieldnames = [
                "title", "info","dateCreated", "ratingValue", "ratingCount", "genres"
            ]
               writer = DictWriter(csv_file, fieldnames=fieldnames)
               write_headers = True

               if write_headers:
                writer.writeheader()  # Write headers only if the file is new

               writer.writerows(data)
         # Check if there's a next page
            next_page_links = driver.find_elements(By.CSS_SELECTOR, '.pagination a #nextpagination')
            next_page_link = next((link for link in next_page_links if 'inactive' not in link.get_attribute('class')), None)

            if next_page_link:
             next_page_number = int(next_page_link.text) if next_page_link.text.isdigit() else 0

            if not next_page_link or current_page_url.endswith(f"page/{next_page_number}"):
               break  # Exit loop if there's no active next page or if it's on the same page

    # Click the next page link using JavaScript
            driver.execute_script("arguments[0].click();", next_page_link)
            time.sleep(2)  # Add a slight delay for the new page to load

    # Wait for the new page to load completely
            WebDriverWait(driver, 10).until(EC.url_changes(current_page_url))
            current_page_url = driver.current_url  # Update the current page URL for the next iteration



    finally:
        # Close the browser
        driver.quit()

# Call the scrape function
scrape()



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Tải dữ liệu
data = pd.read_csv('output.csv')

# Hàm để làm sạch và chuyển đổi cột thành các giá trị số
def clean_numeric_columns(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    return df

# Làm sạch và chuyển đổi các cột 'ratingValue' và 'ratingCount'
data = clean_numeric_columns(data, ['ratingValue', 'ratingCount'])

# Điền giá trị NaN trong các cột 'info' và 'genres' bằng chuỗi trống
data['info'] = data['info'].fillna('')
data['genres'] = data['genres'].fillna('')

# Kết hợp các tính năng liên quan để lọc dữ liệu
data['combined_features'] = data['title'] + ' ' + data['info'] + ' ' + data['ratingValue'].astype(str) + ' ' + data['ratingCount'].astype(str) + ' ' + data['genres']

# Vector hóa TF-IDF cho phân tích văn bản
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

# Chuẩn hóa các tính năng số
scaler = MinMaxScaler()
numerical_features = data[['ratingValue', 'ratingCount']].astype(float)
data[['ratingValue', 'ratingCount']] = scaler.fit_transform(numerical_features)

# Tính ma trận tương đồng cosine
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Hàm để lấy các đề xuất phim dựa trên sự tương đồng
def get_movie_recommendations(movie_title, top_n=5):
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
    ssim_score = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices].tolist()

# Ví dụ: Lấy các đề xuất cho một tựa phim
movie_title = 'Bố Nuôi'
recommendations = get_movie_recommendations(movie_title)
print(f"Các đề xuất cho '{movie_title}': {recommendations}")


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



#Biểu đồ thể loại lấy giá trị đầu
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('output.csv')

# Chuyển cột dateCreated sang định dạng datetime và lọc dữ liệu năm 2023
data['dateCreated'] = pd.to_datetime(data['dateCreated'], errors='coerce')
data_2023 = data[data['dateCreated'].dt.year == 2023]

# Tính số lượng phim theo tháng và thể loại trong năm 2023
monthly_movie_counts = data_2023['dateCreated'].dt.month.value_counts().sort_index()
genres_count = data_2023.groupby(data_2023['dateCreated'].dt.month)['genres'].value_counts().unstack().fillna(0)

# Sắp xếp các cột thể loại theo tổng số lượng phim
genres_count = genres_count[genres_count.sum().sort_values(ascending=False).index]

# Tạo một danh sách các thể loại đã sắp xếp
sorted_genres = genres_count.sum().sort_values(ascending=False).index

# Vẽ biểu đồ
plt.figure(figsize=(12, 8))

# Biểu đồ số lượng phim theo tháng
plt.subplot(2, 1, 1)
monthly_movie_counts.plot(kind='bar', color='skyblue')
plt.title('Số lượng phim theo tháng trong năm 2023')
plt.xlabel('Tháng')
plt.ylabel('Số lượng phim')
plt.xticks(rotation=0)

# Biểu đồ số lượng phim theo thể loại trong từng tháng
plt.subplot(2, 1, 2)
genres_count = genres_count[sorted_genres]  # Áp dụng sắp xếp theo danh sách đã tạo
genres_count.plot(kind='bar', stacked=True)
plt.title('Số lượng phim theo thể loại trong từng tháng (2023)')
plt.xlabel('Tháng')
plt.ylabel('Số lượng phim')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

#Biểu đồ thể loại lấy giá trị đầu
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('output.csv')

# Chuyển cột dateCreated sang định dạng datetime và lọc dữ liệu năm 2023
data['dateCreated'] = pd.to_datetime(data['dateCreated'], errors='coerce')
data_2023 = data[data['dateCreated'].dt.year == 2023]

# Lấy thể loại đầu tiên từ cột genres
data_2023['first_genre'] = data_2023['genres'].str.split(',').str[0]

# Tính số lượng phim theo tháng và thể loại đầu tiên trong năm 2023
monthly_movie_counts = data_2023['dateCreated'].dt.month.value_counts().sort_index()
genre_counts = data_2023.groupby(data_2023['dateCreated'].dt.month)['first_genre'].value_counts().unstack().fillna(0)

# Sắp xếp các cột thể loại theo tổng số lượng phim
genre_counts = genre_counts[genre_counts.sum().sort_values(ascending=False).index]

# Tạo biểu đồ
plt.figure(figsize=(12, 8))

# Biểu đồ số lượng phim theo tháng
plt.subplot(2, 1, 1)
monthly_movie_counts.plot(kind='bar', color='skyblue')
plt.title('Số lượng phim theo tháng trong năm 2023')
plt.xlabel('Tháng')
plt.ylabel('Số lượng phim')
plt.xticks(rotation=0)

# Biểu đồ số lượng phim theo thể loại đầu tiên trong từng tháng
plt.subplot(2, 1, 2)
genre_counts.plot(kind='bar', stacked=True)
plt.title('Số lượng phim theo thể loại đầu tiên trong từng tháng (2023)')
plt.xlabel('Tháng')
plt.ylabel('Số lượng phim')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import networkx as nx
# import matplotlib.pyplot as plt

# # Đọc dữ liệu từ file CSV
# data = pd.read_csv('output.csv')

# # Tạo dataframe chỉ chứa cột 'genres'
# selected_data = data[['genres']]

# # Loại bỏ các dòng có giá trị NaN trong cột 'genres'
# selected_data = selected_data.dropna()

# # Vector hóa dữ liệu 'genres'
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(selected_data['genres'])

# # Áp dụng thuật toán clustering (ví dụ: K-means)
# num_clusters = 5  # Số cụm (clusters) cần phân chia
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# kmeans.fit(X)

# # Thêm nhãn cụm (cluster labels) vào dataframe
# selected_data['cluster'] = kmeans.labels_

# # Tạo đồ thị NetworkX từ dữ liệu phân cụm
# G = nx.Graph()
# for index, row in selected_data.iterrows():
#     G.add_node(row['genres'], cluster=row['cluster'])

# # Sắp xếp các cụm
# pos = nx.spring_layout(G, k=0.5)  # Giảm độ căng thẳng của đồ thị

# # Vẽ đồ thị NetworkX với các cụm tương tự được gom lại
# plt.figure(figsize=(10, 8))
# node_color = [float(G.nodes[node]['cluster']) for node in G]
# nx.draw_networkx(G, pos, node_color=node_color, cmap=plt.cm.Set1, with_labels=True, node_size=300)
# plt.title('Kết quả clustering (Nhóm cụm tương tự)')
# plt.show()


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
