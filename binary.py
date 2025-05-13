import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/ssali/Downloads/archive/tweet_emotions.csv")

sentiment_map = {
    'empty': 0,
    'sadness': 1,
    'enthusiasm': 2,
    'neutral': 3,
    'worry': 4,
}

labels = np.array([sentiment_map.get(label, -1) for label in df['sentiment']])  
df = df[labels != -1]
labels = labels[labels != -1]

sentences = df['content'].values
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {accuracy:.4f}')

model.save('sentiment_model.h5')

def predict_sentiment(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    print(f"Prediction Score: {prediction[0][predicted_class]:.4f}")
    print(f"Emotion: {sentiment}")

while True:
    sentence = input("Enter a sentence to predict (or 'exit' to stop): ")
    if sentence.lower() == 'exit':
        break
    predict_sentiment(sentence)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
