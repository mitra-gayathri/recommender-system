import streamlit as st
import joblib
import pandas as pd

# load saved files
movies = joblib.load("movies_dict.joblib")
similarity = joblib.load("similarity.joblib")

st.title("Movie Recommender System 🎬")

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


selected_movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend top 5 movies"):
    recommendations = recommend(selected_movie_name)
    for movie in recommendations:
        st.write(movie)
