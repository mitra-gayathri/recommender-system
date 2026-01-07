import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Movie Recommender", layout="centered")

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv", engine="python")

# Create tags column (VERY IMPORTANT)
movies['tags'] = (
    movies['overview'].fillna('') + ' ' +
    movies['genres'].fillna('') + ' ' +
    movies['keywords'].fillna('')
)

movies = movies[['title', 'tags']].dropna()

@st.cache_data
def build_similarity(data):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data['tags']).toarray()
    return cosine_similarity(vectors)

similarity = build_similarity(movies)

st.title("Movie Recommender System 🎬")

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    return [movies.iloc[i[0]].title for i in movies_list]

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend top 5 movies"):
    for m in recommend(selected_movie):
        st.write(m)
