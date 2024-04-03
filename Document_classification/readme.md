# Document Classification App

## Introduction
This project develops a Document Classification application using machine learning and natural language processing (NLP) techniques. The app classifies input text into predefined categories, leveraging the 20 Newsgroups dataset. It demonstrates the power of text classification using both traditional machine learning models, like Naive Bayes, and deep learning models, such as Recurrent Neural Networks (RNN).

## Project Structure
The project is organized into three main files:
- `datapreprocessing_file.py`: Contains code for data preprocessing, including tokenization, removing stopwords, stemming, and vectorization of text data.
- `model_training.py`: Includes the machine learning pipeline for training Naive Bayes and RNN models, evaluating their performance, and saving the trained models.
- `user_interface.py`: A Streamlit-based web application that allows users to input text and see the predicted category by the trained model.

## Approach and Methodology
### Data Preprocessing
- **Tokenization:** Splitting text into individual words.
- **Stopwords Removal:** Eliminating common words that add little value in the context of text classification.
- **Stemming:** Reducing words to their root form.
- **Vectorization (TF-IDF):** Transforming text into a meaningful vector of numbers.

### Model Training
- **Naive Bayes Classifier:** A probabilistic model ideal for text classification due to its simplicity and effectiveness.
- **Recurrent Neural Network (RNN):** Utilizes LSTM layers to capture the sequence and context of words in text data, providing a more nuanced understanding of the content.

### Web Application
Built using Streamlit, the app offers a user-friendly interface for real-world application of the trained models. Users can input text and receive instant classification results, demonstrating the practical use of NLP models.

## Installation and Setup
Ensure Python 3.6+ is installed on your system. Follow these steps to set up the project environment:


## Reproducing the Results
1. Clone the repository:
   ```bash
   git clone
    ```
2. Navigate to the project directory:
    ```bash
    cd Document_classification
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run user_interface.py/"Path to the file"
    ```

## Usage and Examples

1. Usage Examples
Below are examples that highlight the model's capability to classify texts accurately:

    Input Text: "Exploring the vast universe, NASA continues to make significant advancements in space technology."
    Predicted Category: sci.space

    Input Text: "The latest advancements in computer graphics have revolutionized game design, offering unprecedented realism."
    Predicted Category: comp.graphics

    Input Text: "Understanding the complexities of the political landscape requires a deep analysis of historical and current events."
    Predicted Category: talk.politics.misc