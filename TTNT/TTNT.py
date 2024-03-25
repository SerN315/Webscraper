import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import tkinter as tk
from tkinter import messagebox

# Read event data from CSV file
events_data = pd.read_csv('events.csv')

# Preprocess data for LDA model
vectorizer = TfidfVectorizer(stop_words='english')
events_text = events_data['description'].values.astype('U')
events_text_vectorized = vectorizer.fit_transform(events_text)

# Train LDA model
num_topics = events_data['topic'].nunique()
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(events_text_vectorized)

# Preprocess data for RNN model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(events_text)
sequences = tokenizer.texts_to_sequences(events_text)
max_sequence_length = max([len(seq) for seq in sequences])
events_text_padded = pad_sequences(sequences, maxlen=max_sequence_length)

# Train RNN model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

rnn_model = Sequential()
rnn_model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
rnn_model.add(LSTM(256))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(num_topics, activation='softmax'))
rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

target = pd.get_dummies(events_data['topic']).values
train_indices = np.random.rand(len(events_data)) < 0.8
train_text = events_text_padded[train_indices]
train_target = target[train_indices]
test_text = events_text_padded[~train_indices]
test_target = target[~train_indices]

rnn_model.fit(train_text, train_target, validation_data=(test_text, test_target), epochs=20, batch_size=32)

def get_event_recommendations(event_description):
    event_text_vectorized = vectorizer.transform([event_description])
    event_topic = lda_model.transform(event_text_vectorized)[0]

    event_sequence = tokenizer.texts_to_sequences([event_description])
    event_sequence_padded = pad_sequences(event_sequence, maxlen=max_sequence_length)
    event_topic_prediction = rnn_model.predict(event_sequence_padded)[0]

    topic_indices = np.argsort(event_topic)[::-1][:3]
    valid_topic_indices = events_data['topic'].unique()
    valid_topic_indices.sort()
    valid_topic_indices = valid_topic_indices.astype(int)
    valid_topic_indices = valid_topic_indices.tolist()
    topic_indices = [topic for topic in topic_indices if topic in valid_topic_indices]

    topic_events = events_data[events_data['topic'].isin(topic_indices)]
    topic_events = topic_events.reset_index(drop=True)

    if not topic_events.empty:
        topic_events['topic_prediction'] = event_topic_prediction[topic_events.index]
        topic_events['topic_description'] = events_data['topic_des'][topic_events.index]
        topic_events = topic_events.sort_values(by='topic_prediction', ascending=False).head(3)

    return topic_events

# Create a Tkinter window
window = tk.Tk()
window.title("Event Recommendations")

# Create a text box to enter the event description
description_label = tk.Label(window, text="Event Description:")
description_label.pack()

description_entry = tk.Entry(window, width=50)
description_entry.pack()

# Create a button to get recommendations
def get_recommendations():
    event_description = description_entry.get()
    recommendations = get_event_recommendations(event_description)

    if recommendations.empty:
        messagebox.showinfo("Event Recommendations", "No recommendations found.")
    else:
        recommendation_text = "Recommendations:\n\n"
        for index, row in recommendations.iterrows():
            recommendation_text += f"Event: {row['eventname']}\n"
            recommendation_text += f"Topic: {row['topic_des']}\n"
            recommendation_text += f"Description: {row['description']}\n"
            recommendation_text += "---"
        messagebox.showinfo("Event Recommendations", recommendation_text)

# Create a button to trigger the recommendation process
recommend_button = tk.Button(window, text="Get Recommendations", command=get_recommendations)
recommend_button.pack()

# Run the Tkinter event loop
window.mainloop()