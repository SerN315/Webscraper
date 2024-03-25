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
def train_lda_model(data):
    # Preprocess text for LDA (tokenization, removing stopwords, stemming, etc.)
    # Then train LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)
    return lda

# Combine RNN and LDA representations
def combine_representations(rnn_model, lda_model, tfidf_matrix):
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
