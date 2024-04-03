import pandas as pd

def preprocess_data(data_path):
  df = pd.read_csv(data_path)
  df_copy = df.copy()
  df_copy.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Summary'],
                inplace=True, axis=1)
  df_copy["Sentiment"] = df_copy["Score"].apply(lambda score: "positive" if score >= 3 else ("negative" if score <= 2 else "neutral"))
  df_copy['Sentiment'] = df_copy['Sentiment'].map({'positive':1, 'negative':0, 'neutral':2})
  data = df_copy[["Sentiment", "Text"]]
  return data

if __name__ == "__main__":
  data = preprocess_data("Reviews.csv")
  data.to_csv("preprocessed_data.csv", index=False)

#For creating model and preparing data for training and testing refer to the following code:

# from sklearn.model_selection import train_test_split
# from transformers import DistilBertTokenizerFast
# from stop_words import get_stop_words

# def load_dataset(file_path):
#   """
#   Load the dataset from a CSV file.
#   """
#   df = pd.read_csv(file_path)
#   return df

# def preprocess_data(df, stop_words_language='english'):
#   """
#   Perform preprocessing on the loaded dataset, including stop word removal.
#   """
#   df_copy = df.copy()
#   df_copy.drop(['Id', 'ProductId', 'UserId', 'ProfileName',
#                 'HelpfulnessNumerator', 'HelpfulnessDenominator',
#                 'Time', 'Summary'], inplace=True, axis=1)

#   stop_words = get_stop_words(stop_words_language)

#   def clean_text(text):
#     # Convert to lowercase, remove punctuation, and remove stop words
#     text = text.lower()
#     text = ''.join([char for char in text if char.isalnum() or char == ' '])
#     words = [word for word in text.split() if word not in stop_words]
#     return ' '.join(words)

#   df_copy["Text"] = df_copy["Text"].apply(clean_text)

#   df_copy["Sentiment"] = df_copy["Score"].apply(lambda score: "positive" if score >= 3 else ("negative" if score <= 2 else "neutral"))
#   df_copy['Sentiment'] = df_copy['Sentiment'].map({'positive':1, 'negative':0, 'neutral':2})

#   return df_copy

# def prepare_data(df, num_samples=1000, test_size=0.2):
#   """
#   Prepare data for training and testing.
#   """
#   data = df[["Sentiment", "Text"]].head(num_samples)
#   reviews = data['Text'].values.tolist()
#   labels = data['Sentiment'].tolist()

#   training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(reviews, labels, test_size=test_size)

#   return training_sentences, validation_sentences, training_labels, validation_labels

# def tokenize_sentences(sentences, tokenizer, max_length=128):
#   """
#   Tokenize input sentences using the specified tokenizer.
#   """
#   encodings = tokenizer(sentences, truncation=True, padding=True, max_length=max_length)
#   return encodings

# if __name__ == "__main__":
#   # Example usage
#   file_path = 'Reviews.csv'
#   tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#   # Load dataset
#   df = load_dataset(file_path)

#   # Preprocess data
#   preprocessed_df = preprocess_data(df)

#   # Prepare data for training and testing
#   training_sentences, validation_sentences, training_labels, validation_labels = prepare_data(preprocessed_df)

#   # Tokenize sentences
#   train_encodings = tokenize_sentences(training_sentences, tokenizer)
#   val_encodings = tokenize_sentences(validation_sentences, tokenizer)
