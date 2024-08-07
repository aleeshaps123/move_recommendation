import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained model and encoder
model_file_path = 'best_model.pkl'
encoder_file_path = 'encoder.pkl'
with open(model_file_path, 'rb') as model_file:
    best_algo = pickle.load(model_file)
with open(encoder_file_path, 'rb') as encoder_file:
    ohe = pickle.load(encoder_file)

# Load the dataset for recommendation lookup
file_path = 'imdb_top_movies.csv'
movies_df = pd.read_csv(file_path)

# Inject CSS directly in the code
css = """
<style>
body {
    background-color: #f0f2f6;
}

.stApp {
    display: flex;
    justify-content: center;
    align-items: center;
}

.stApp > div:nth-child(1) {
    flex: 1;
}

.stApp > div:nth-child(2) {
    flex: 2;
    background-image: url('https://motionarray.imgix.net/preview-86762-6dicuHGhCw-high_0014.jpg?w=660&q=60&fit=max&auto=format'); /* Change to your background image URL */
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    border-radius: 10px;
    padding: 20px;
    color: white;
    text-shadow: 1px 1px 2px #000000;
}

.sidebar .sidebar-content {
    background: rgba(255, 255, 255, 0.8); /* Semi-transparent background for the sidebar */
    padding: 20px;
    border-radius: 10px;
}

.stButton > button {
    width: 100%;
    padding: 10px;
    font-size: 16px;
    border-radius: 5px;
    background-color: #FF4B4B; /* Background color */
    color: white; /* Text color */
    border: none; /* Remove border */
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Add shadow */
    transition: all 0.3s ease; /* Smooth transition */
}

.stButton > button:hover {
    background-color: #FF6B6B; /* Darker background on hover */
    box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.15); /* Bigger shadow on hover */
}

.custom-title {
    color: white; /* Set the title color to white */
    text-shadow: 2px 2px 4px #000000; /* Add text shadow */
    font-size: 36px; /* Adjust font size if needed */
    text-align: center; /* Center align the title */
    margin-top: 20px; /* Add some top margin */
}

.movie-box {
    background-color: #ffffff; /* White background for the movie box */
    color: #000000; /* Black text color */
    border-radius: 10px; /* Rounded corners */
    padding: 15px; /* Padding inside the box */
    margin-bottom: 15px; /* Margin between boxes */
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3); /* Shadow around the box */
}

.st-subheader {
    color: #ffffff; /* White color for the subheader text */
    text-shadow: 2px 2px 4px #000000; /* Add text shadow */
}

.stText {
    color: #ffffff;
    text-shadow: 1px 1px 2px #000000;
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Use custom HTML for the title
st.markdown("<h1 class='custom-title'>Movie Recommendation System</h1>", unsafe_allow_html=True)

st.sidebar.header("User Input")

# Input fields
imdb_rating = st.sidebar.slider("IMDB Rating", min_value=0.0, max_value=10.0, step=0.1)
genre = st.sidebar.text_input("Genre", value='Crime, Drama')
director = st.sidebar.text_input("Director", value='Francis Ford Coppola')
cast = st.sidebar.text_input("Cast", value='Al Pacino')

def recommend_movies(imdb_rating, genre, director, cast, top_n=10):
    input_data = pd.DataFrame({
        'IMDB Rating': [imdb_rating],
        'Genre': [genre],
        'Director': [director],
        'Cast': [cast]
    })
    input_encoded = ohe.transform(input_data[['Genre', 'Director', 'Cast']]).toarray()
    input_combined = pd.concat([pd.DataFrame(input_data['IMDB Rating']), pd.DataFrame(input_encoded)], axis=1)
    input_combined.columns = input_combined.columns.astype(str)  # Ensure all column names are strings

    # Predict movie names
    movie_predictions = best_algo.predict(input_combined)
    movie_scores = best_algo.predict_proba(input_combined)[0]

    # Create a DataFrame with movies and their predicted scores
    movie_score_df = pd.DataFrame({
        'Movie Name': best_algo.classes_,
        'Score': movie_scores
    })

    # Sort movies by score and get the top N
    recommended_movies = movie_score_df.sort_values(by='Score', ascending=False).head(top_n)

    # Retrieve movie details for the recommended movies
    recommendations = []
    for movie in recommended_movies['Movie Name']:
        movie_details = movies_df[movies_df['Movie Name'] == movie].iloc[0]
        recommendations.append({
            'Movie Name': movie,
            'IMDB Rating': movie_details['IMDB Rating'],
            'Genre': movie_details['Genre'],
            'Cast': movie_details['Cast'],
            'Overview': movie_details['Overview']
        })
    
    return recommendations

if st.sidebar.button("Get Recommendations"):
    results = recommend_movies(imdb_rating, genre, director, cast)
    st.subheader("Recommended Movies")
    for result in results:
        st.markdown(f"""
        <div class='movie-box'>
            <h3>{result['Movie Name']}</h3>
            <p><strong>IMDB Rating:</strong> {result['IMDB Rating']}</p>
            <p><strong>Genre:</strong> {result['Genre']}</p>
            <p><strong>Cast:</strong> {result['Cast']}</p>
            <p><strong>Overview:</strong> {result['Overview']}</p>
        </div>
        """, unsafe_allow_html=True)
