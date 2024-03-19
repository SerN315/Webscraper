import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Đọc dữ liệu sự kiện từ file CSV
events_data = pd.read_csv('events.csv')

# Tiền xử lý dữ liệu cho mô hình LDA
vectorizer = CountVectorizer(stop_words='english')
events_text = events_data['description'].values.astype('U')
events_text_vectorized = vectorizer.fit_transform(events_text)

# Huấn luyện mô hình LDA
num_topics = 10 # Số lượng chủ đề
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(events_text_vectorized)

# Tiền xử lý dữ liệu cho mô hình RNN
tokenizer = Tokenizer()
tokenizer.fit_on_texts(events_text)
sequences = tokenizer.texts_to_sequences(events_text)
max_sequence_length = max([len(seq) for seq in sequences])
events_text_padded = pad_sequences(sequences, maxlen=max_sequence_length)

# Xây dựng mô hình RNN
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
rnn_model = Sequential()
rnn_model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
rnn_model.add(LSTM(128))
rnn_model.add(Dense(num_topics, activation='softmax'))
rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Chuẩn bị dữ liệu huấn luyện cho mô hình RNN
target = pd.get_dummies(events_data['topic']).values
train_indices = np.random.rand(len(events_data)) < 0.8
train_text = events_text_padded[train_indices]
train_target = target[train_indices]
test_text = events_text_padded[~train_indices]
test_target = target[~train_indices]

# Huấn luyện mô hình RNN
rnn_model.fit(train_text, train_target, validation_data=(test_text, test_target), epochs=30, batch_size=32)


# Gợi ý sự kiện dựa trên mô hình đã huấn luyện
def get_event_recommendations(event_description):
    # Gợi ý chủ đề sự kiện bằng mô hình LDA
    event_text_vectorized = vectorizer.transform([event_description])
    event_topic = lda_model.transform(event_text_vectorized)[0]

    # Gợi ý sự kiện cụ thể bằng mô hình RNN
    event_sequence = tokenizer.texts_to_sequences([event_description])
    event_sequence_padded = pad_sequences(event_sequence, maxlen=max_sequence_length)
    event_topic_prediction = rnn_model.predict(event_sequence_padded)[0]

    # Lấy các sự kiện gần nhất với chủ đề và dự đoán từ mô hình RNN
    topic_indices = np.argsort(event_topic)[::-1][:3]
    topic_events = events_data[events_data['topic'].isin(topic_indices)]
    rnn_event_indices = np.argsort(event_topic_prediction)[::-1][:min(3,len(topic_events))]
    rnn_event_indices = rnn_event_indices[rnn_event_indices < len(topic_events)]
    rnn_events = topic_events.iloc[rnn_event_indices]

    return rnn_events

# Sử dụng mô hình để gợi ý sự kiện
event_description = "CYBER SPACE IS THE BEST"
recommendations = get_event_recommendations(event_description)
print(recommendations)


