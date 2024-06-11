import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

print("Script started")

# Load the raw wine dataset
wine_data = pd.read_csv('raw_wine_data.csv')
print("Data loaded")

# Text preprocessing function
def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Remove punctuation and lower the text
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

# Preprocess the description column
wine_data['processed_description'] = wine_data['description'].apply(preprocess_text)
print("Text preprocessing done")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(wine_data['processed_description'])
print("TF-IDF vectorization done")

# Save the processed data and vectorizer
wine_data.to_csv('processed_wine_data.csv', index=False)
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
print("Data and model saved")

print("Script finished")
