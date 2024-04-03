import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def extract_features():
    # Load the 20 Newsgroups dataset
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data
    labels = newsgroups.target

    # Preprocess the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(docs)

    # Encode the labels
    label_encoder = label_encoder()
    y = label_encoder.fit_transform(labels)

    return X, y