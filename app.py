import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Ensure necessary NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load the processed data and TF-IDF vectorizer
wine_data = pd.read_csv('processed_wine_data.csv')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')

# Sidebar for user inputs
st.sidebar.title("Wine Recommendation")
wine_name = st.sidebar.text_input("Enter a wine you like")
wine_type = st.sidebar.selectbox("Select Wine Type", options=['Any', 'Red', 'White', 'Rose'])
country = st.sidebar.selectbox("Select Country", options=['Any'] + wine_data['country'].unique().tolist())
region = st.sidebar.selectbox("Select Region", options=['Any'] + wine_data['region'].unique().tolist())
price_range = st.sidebar.selectbox("Select Price Range", options=['Any', '<$20', '$20-$50', '>$50'])
rating = st.sidebar.slider("Minimum Rating", min_value=80, max_value=100, value=85)
description = st.sidebar.text_area("Describe the type of wine you like")

# Filter the dataset based on user inputs
filtered_data = wine_data[
    ((wine_data['title'].str.contains(wine_name, case=False)) if wine_name else True) &
    ((wine_data['variety'] == wine_type) if wine_type != 'Any' else True) &
    ((wine_data['country'] == country) if country != 'Any' else True) &
    ((wine_data['region'].str.contains(region)) if region != 'Any' else True) &
    ((wine_data['price'].between(0, 20)) if price_range == '<$20' else True) &
    ((wine_data['price'].between(20, 50)) if price_range == '$20-$50' else True) &
    ((wine_data['price'] > 50) if price_range == '>$50' else True) &
    (wine_data['points'] >= rating)
]

# Text preprocessing function for user input
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

# Process user description and transform it using the same vectorizer
user_desc_processed = preprocess_text(description)
user_tfidf = vectorizer.transform([user_desc_processed])

# Calculate similarity scores
cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
wine_data['similarity_score'] = cosine_similarities

# Sort the wines based on similarity score and other filters
recommended_wines = wine_data[
    ((wine_data['variety'] == wine_type) if wine_type != 'Any' else True) &
    ((wine_data['country'] == country) if country != 'Any' else True) &
    ((wine_data['region'].str.contains(region)) if region != 'Any' else True) &
    ((wine_data['price'].between(0, 20)) if price_range == '<$20' else True) &
    ((wine_data['price'].between(20, 50)) if price_range == '$20-$50' else True) &
    ((wine_data['price'] > 50) if price_range == '>$50' else True) &
    (wine_data['points'] >= rating)
].sort_values(by='similarity_score', ascending=False)

# Display recommendations
st.title("Recommended Wines")
for index, row in recommended_wines.iterrows():
    st.subheader(f"{row['title']} ({row['points']} points)")
    st.write(f"Type: {row['variety']}")
    st.write(f"Country: {row['country']}")
    st.write(f"Region: {row['region']}")
    st.write(f"Price: ${row['price']}")
    st.write(f"Tasting Notes: {row['description']}")
    st.write("---")