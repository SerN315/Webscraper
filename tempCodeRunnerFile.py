import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('output.csv')

# Combine relevant features for content-based filtering
data['combined_features'] = data['genres'] + ' ' + data['ratingValue'] + ' ' + data['ratingCount'] + ' ' + data['info']

# TF-IDF Vectorization for textual analysis
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

# Normalize numerical features
scaler = MinMaxScaler()
numerical_features = data[['ratingValue', 'ratingCount']].astype(float)
data[['ratingValue', 'ratingCount']] = scaler.fit_transform(numerical_features)

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on similarity
def get_movie_recommendations(movie_title, top_n=5):
    idx = data[data['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]

# Example: Get recommendations for a movie title
movie_title = 'Your Movie Title'
recommendations = get_movie_recommendations(movie_title)
print(f"Recommendations for '{movie_title}': {recommendations}")