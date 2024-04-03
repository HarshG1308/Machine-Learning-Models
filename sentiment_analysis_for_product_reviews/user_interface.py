# Description: This script contains the code to create a simple user interface for the sentiment analysis model. (using Streamlit)

import streamlit as st
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf

# Load the pre-trained tokenizer and model (assuming they are saved in a folder named "sentiment")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained(
    r"C:\Users\Asus\Desktop\Submission\sentiment\tf_model.h5",
    config=r"C:\Users\Asus\Desktop\Submission\sentiment\config.json"
)

def predict_sentiment(text):
  # Preprocess the input text (replace with your actual preprocessing steps)
  processed_text = text.strip()

  # Tokenize the text
  inputs = tokenizer(processed_text, truncation=True, padding=True, return_tensors="tf")

  # Get the model prediction
  outputs = model(inputs)[0]
  predictions = tf.nn.softmax(outputs, axis=1)
  label = tf.argmax(predictions, axis=1)
  label = label.numpy()[0]

  # Map the label to sentiment text
  sentiment = ["Negative", "Positive", "Neutral"][label]
  return sentiment

st.title("Sentiment Analysis App")

# Text area for user input
user_input = st.text_area("Enter your review:")

# Button to trigger prediction
if st.button("Predict"):
  prediction = predict_sentiment(user_input)
  st.write(f"Sentiment: {prediction}")

#Run the app using: streamlit run sentiment_analysis_for_product_reviews\user_interface.py



# Description: This script contains the code to create a simple user interface for the sentiment analysis model. (using tkinter)

# from tkinter import Tk, Label, Entry, Button
# from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
# import tensorflow as tf

# # Load the pre-trained tokenizer and model (assuming they are saved in a folder named "sentiment")
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# model = TFDistilBertForSequenceClassification.from_pretrained(
#     r".\sentiment\tf_model.h5",
#     config=r".\sentiment\config.json"
# )

# def predict_sentiment(text):
#   # Preprocess the input text (replace with your actual preprocessing steps)
#   processed_text = text.strip()

#   # Tokenize the text
#   inputs = tokenizer(processed_text, truncation=True, padding=True, return_tensors="tf")

#   # Get the model prediction
#   outputs = model(inputs)[0]
#   predictions = tf.nn.softmax(outputs, axis=1)
#   label = tf.argmax(predictions, axis=1)
#   label = label.numpy()[0]

#   # Map the label to sentiment text
#   sentiment = ["Negative", "Positive", "Neutral"][label]
#   return sentiment

# def handle_prediction():
#   text = entry.get()
#   prediction = predict_sentiment(text)
#   result_label.config(text=f"Sentiment: {prediction}")

# # Create the Tkinter app
# root = Tk()
# root.title("Sentiment Analysis App")

# # Label for user input
# label = Label(root, text="Enter your review:")
# label.pack()

# # Entry field for user input
# entry = Entry(root)
# entry.pack()

# # Button to trigger prediction
# button = Button(root, text="Predict", command=handle_prediction)
# button.pack()

# # Label to display the prediction result
# result_label = Label(root, text="")
# result_label.pack()

# root.mainloop()

