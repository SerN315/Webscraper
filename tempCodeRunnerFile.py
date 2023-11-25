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
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices].tolist()

# Ví dụ: Lấy các đề xuất cho một tựa phim
movie_title = 'Bố Nuôi'
recommendations = get_movie_recommendations(movie_title)
print(f"Các đề xuất cho '{movie_title}': {recommendations}")