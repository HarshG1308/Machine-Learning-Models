from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf

class SentimentModel:

  def __init__(self, model_path):
    self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    self.model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

  def predict(self, text):
    processed_text = text.strip()

    # Tokenize the text
    inputs = self.tokenizer(processed_text, truncation=True, padding=True, return_tensors="tf")

    # Get the model prediction
    outputs = self.model(inputs)[0]
    predictions = tf.nn.softmax(outputs, axis=1)
    label = tf.argmax(predictions, axis=1)
    label = label.numpy()[0]

    # Map the label to sentiment text
    sentiment = ["Negative", "Positive", "Neutral"][label]
    return sentiment

if __name__ == "__main__":
  model_path = r"C:\Users\Asus\Desktop\Submission\sentiment\tf_model.h5"
  model = SentimentModel(model_path)

  text = "This movie was truly awful."
  prediction = model.predict(text)
  print(f"Predicted sentiment: {prediction}")


# def build_model(num_labels=2):
#   """
#   Build the NLP model using DistilBERT for sequence classification.
#   """
#   model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
#   return model

# def compile_model(model, learning_rate=5e-5, epsilon=1e-08):
#   """
#   Compile the model with optimizer and loss function.
#   """
#   optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
#   model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])

# def train_model(model, train_dataset, val_dataset, epochs=3, batch_size=16):
#   """
#   Train the NLP model.
#   """
#   model.fit(train_dataset.shuffle(100).batch(batch_size),
#             epochs=epochs,
#             validation_data=val_dataset.shuffle(100).batch(batch_size))

# def save_model(model, save_path):
#   """
#   Save the trained model to the specified path.
#   """
#   model.save_pretrained(save_path)

# if __name__ == "__main__":
#   # Example usage
#   num_labels = 2
#   model = build_model(num_labels)

#   # Prepare train and validation datasets (already prepared in preprocessing)
#   # train_dataset and val_dataset are assumed to be available here

#   # Compile the model
#   compile_model(model)

#   # Train the model
#   train_model(model, train_dataset, val_dataset)

#   # Save the trained model
#   save_path = "./sentiment"
#   save_model(model, save_path)