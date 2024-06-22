import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Download the dataset from https://www.kaggle.com/tmdb/tmdb-movie-metadata 
# and place it in the same directory as this script.

# Load data function with st.cache_data
@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
def load_data():
    # Load the credits and movies datasets
    credits = pd.read_csv(r'C:\Users\aaron\OneDrive\Desktop\projects\recomender system\data\tmdb_5000_credits.csv')
    movies = pd.read_csv(r'C:\Users\aaron\OneDrive\Desktop\projects\recomender system\data\tmdb_5000_movies.csv')
    return credits, movies

# Function to preprocess and stem tags
def preprocess_tags(text):
    # Initialize Porter Stemmer for stemming words
    ps = PorterStemmer()
    # Apply stemming to each word in the text
    return " ".join([ps.stem(word) for word in text.split()])

# Process data function with st.cache_data
@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
def process_data(data):
    # Iterate over specified columns and preprocess each one
    for column in ['overview', 'keywords', 'genres', 'cast', 'crew']:
        data[column] = data[column].apply(lambda x: preprocess_tags(str(x)))
    
    # Combine relevant columns into 'tags' column
    data['tags'] = data['overview'] + ' ' + data['keywords'] + ' ' + data['genres'] + ' ' + data['cast'] + ' ' + data['crew']
    
    # Vectorize 'tags' column using CountVectorizer
    vectorizer = CountVectorizer()
    tag_matrix = vectorizer.fit_transform(data['tags'])
    
    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(tag_matrix)
    
    return data, cosine_sim

# Streamlit app
def main():
    st.title('Movie Recommendation System')
    
    # Load data
    credits, movies = load_data()
    
    # Merge credits and movies datasets on the 'title' column
    data = movies.merge(credits, on='title')
    
    # Select features
    features = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
    data = data[features]
    
    # Process data (vectorize and compute cosine similarity)
    data, cosine_sim = process_data(data)
    
    # Function to recommend movies
    def recommend(movie_title):
        # Find the index of the movie in the dataframe that matches the input movie_title (case insensitive)
        idx = data[data['title'].str.lower() == movie_title.lower()].index[0]
        
        # Get cosine similarity scores for the movie at index 'idx' with all other movies
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort movies based on similarity scores in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude the first movie (itself) and select the top 5 similar movies
        sim_scores = sim_scores[1:6]
        
        # Extract indices of the top 5 similar movies
        movie_indices = [i[0] for i in sim_scores]
        
        # Return titles of the top 5 similar movies as a list
        return data.iloc[movie_indices]['title'].values.tolist()
    
    # Streamlit interface
    movie_title = st.text_input('Enter a movie title', 'Avatar')
    if st.button('Recommend'):
        recommended_movies = recommend(movie_title)
        st.subheader('Recommended Movies:')
        for i, movie in enumerate(recommended_movies):
            st.write(f"{i+1}. {movie}")

if __name__ == '__main__':
    main()
