Spam Classifier
This project builds a Spam Classifier using Naive Bayes and TF-IDF Vectorization to classify email messages into spam and ham categories. The classifier is trained on a balanced dataset and can be used to predict whether a message is spam or not.

Project Overview
This repository contains the code to:

Preprocess the text data.

Train a Multinomial Naive Bayes model on the dataset.

Save the trained model and vectorizer for future use.

Requirements
The following Python libraries are required to run this project:

numpy

pandas

scikit-learn

re

string

pickle

You can install the necessary dependencies using pip:

bash
Copy
Edit
pip install numpy pandas scikit-learn
Dataset
The dataset used for this project is the Spam.csv file, which contains a collection of spam and ham (non-spam) messages. The dataset is loaded, preprocessed, and balanced (ensuring an equal number of spam and ham messages).

Data Columns:
Label: The label for the message, either spam or ham.

Text: The content of the message.

File Descriptions
spam_classifier.py
This script includes all necessary steps for preprocessing the dataset, training the classifier, evaluating it, and saving the model.

spam_classifier.pkl
The saved trained Naive Bayes model that can be used for predicting new messages as spam or ham.

vectorizer.pkl
The saved TF-IDF Vectorizer used to transform text data into a numerical format before classification.

Spam.csv
The CSV file containing the dataset with columns Label (spam/ham) and Text.

How to Use
1. Loading the Model and Vectorizer
You can load the trained model and vectorizer like this:

python
Copy
Edit
import pickle

# Load the model
with open("spam_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
2. Predicting Spam or Ham
You can use the model to predict whether a message is spam or ham:

python
Copy
Edit
def predict_spam_or_ham(message):
    # Preprocess the message
    clean_message = preprocess_text(message)
    
    # Vectorize the message
    vectorized_message = vectorizer.transform([clean_message])
    
    # Make a prediction
    prediction = model.predict(vectorized_message)
    
    return prediction[0]


prediction = predict_spam_or_ham(message)
print(f"The message is: {prediction}")
3. Model Evaluation
To evaluate the performance of the model on a test set, the script prints a classification report that includes metrics like precision, recall, and F1-score.

License
This project is licensed under the MIT License.

