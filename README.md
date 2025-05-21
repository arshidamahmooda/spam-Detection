#  Spam vs. Ham Email Classifier

This project is a **machine learning application** that classifies email messages as either **spam** or **ham** (non-spam) using a **Multinomial Naive Bayes classifier** with **TF-IDF vectorization**. 

##  Features

- Binary classification: **Spam** or **Ham**
- Text preprocessing: lowercasing, punctuation and digit removal
- TF-IDF vectorization of email content
- Naive Bayes model for classification
- Balanced dataset for fair model training
- Simple and interactive **Streamlit** user interface
- Downloadable CSV results (optional)
- Easily deployable on Streamlit Cloud
- 
## Model Overview

- **Vectorizer**: `TfidfVectorizer`
- **Classifier**: `MultinomialNB`
- **Evaluation**: Accuracy, Precision, Recall, F1-score
- **Data Source**: [SMS Spam Collection Dataset (UCI)] local CSV

##  Project Structure

spam-detector/
├── Spam (2).csv # Dataset (must be present locally or downloaded)
├── train_model.py # Script to train and save the model
├── streamlit_app.py # Streamlit frontend for classification
├── spam_classifier.pkl # Trained model (generated after training)
├── requirements.txt # Project dependencies
├── README.md # Project documentation


## ⚙️ Installation & Setup

1.   Clone the repository
2. Install dependencies
pip install -r requirements.txt
3. Train the model (optional)
python train_model.py
This will generate spam_classifier.pkl.
4. Run the Streamlit app

