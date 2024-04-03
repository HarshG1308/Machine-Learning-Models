from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from data_preprocessing import preprocess_text
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']


# Using Naive Bayes for text classification
train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
predicted_labels = model.predict(test.data)

# Calculate accuracy
accuracy = accuracy_score(test.target, predicted_labels)
accuracy = accuracy * 100
print(f"Accuracy of the Naive Bayes model: {accuracy:.2f}%")
# Save the model to a file
joblib.dump(model, 'text_classification_model.pkl')


def predict_category(s):
    pred = model.predict([s])
    predicted_category = model.named_steps['multinomialnb'].classes_[pred][0]
    return categories[predicted_category]

# Using RNN for text classification

# Load the 20 newsgroups dataset
# categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
#               'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 
#               'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 
#               'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 
#               'sci.space', 'soc.religion.christian', 'talk.politics.guns', 
#               'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

# train_data = fetch_20newsgroups(subset='train', categories=categories)
# test_data = fetch_20newsgroups(subset='test', categories=categories)

# # Tokenize the texts
# max_words = 10000  # consider only the top 10,000 words
# tokenizer = Tokenizer(num_words=max_words)
# tokenizer.fit_on_texts(train_data.data)

# # Convert texts to sequences
# train_sequences = tokenizer.texts_to_sequences(train_data.data)
# test_sequences = tokenizer.texts_to_sequences(test_data.data)

# # Pad sequences to ensure uniform length
# max_sequence_length = 200  # choose a suitable sequence length
# X_train = pad_sequences(train_sequences, maxlen=max_sequence_length)
# X_test = pad_sequences(test_sequences, maxlen=max_sequence_length)

# # Convert target labels to one-hot encoding
# num_classes = len(categories)
# y_train = np.zeros((len(train_data.target), num_classes))
# y_train[np.arange(len(train_data.target)), train_data.target] = 1

# y_test = np.zeros((len(test_data.target), num_classes))
# y_test[np.arange(len(test_data.target)), test_data.target] = 1

# # Define the RNN model
# embedding_dim = 100  # dimension of word embeddings
# model = Sequential()
# model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
# model.add(LSTM(units=128))
# model.add(Dense(units=num_classes, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# batch_size = 128
# epochs = 5
# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# # text = "The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as"
# # predict_category(text)

# def RNNpredict_category(new_text):
#     new_sequence = tokenizer.texts_to_sequences([new_text])
#     new_padded_sequence = pad_sequences(new_sequence, maxlen=max_sequence_length)

#     # Make predictions using the trained model
#     predicted_probs = model.predict(new_padded_sequence)[0]  # Get the predicted probabilities for each category

#     # Find the index of the category with the highest probability
#     predicted_category_index = np.argmax(predicted_probs)

#     # Get the name of the predicted category
#     predicted_category = categories[predicted_category_index]

#     print(f"Predicted Category: {predicted_category}")
#     # Tokenize and pad the new text
