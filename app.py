import streamlit as st
import pandas as pd
import requests
from io import StringIO
from collections import defaultdict
import numpy as np

# Load the cleaned data
csv_url = ("cleaned_courses")

response = requests.get(csv_url)
response.raise_for_status()  # Ensure the request was successful

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(StringIO(response.text))

# Precompute Language Models for each document (course)
def build_language_model(corpus):
    models = []
    for doc in corpus:
        term_freq = defaultdict(int)
        words = doc.split()
        for word in words:
            term_freq[word] += 1
        total_words = len(words)
        model = {word: freq / total_words for word, freq in term_freq.items()}
        models.append(model)
    return models

df['combined_features'] = df['combined_features'].fillna('').str.lower()  # Convert course data to lowercase
language_models = build_language_model(df['combined_features'])

# Smoothing function (e.g., Laplace smoothing)
def smooth_probability(prob, vocab_size):
    return (prob + 1) / (vocab_size + 1)

# Function to calculate query likelihood score
def query_likelihood(query, language_models, vocab_size):
    query_words = query.lower().split()  # Convert query to lowercase
    scores = []
    for model in language_models:
        score = 1
        for word in query_words:
            word_prob = model.get(word, 0)
            score *= smooth_probability(word_prob, vocab_size)
        scores.append(score)
    return np.array(scores)

# Function to recommend courses based on user query using QLM
def recommend_courses_qlm(query, difficulty=None, min_rating=0):
    vocab_size = len(set(" ".join(df['combined_features']).split()))
    scores = query_likelihood(query, language_models, vocab_size)
    df['Score'] = scores
    recommendations = df.sort_values('Score', ascending=False)

    if difficulty:
        recommendations = recommendations[recommendations['Difficulty'] == difficulty]
    recommendations = recommendations[recommendations['Ratings'] >= min_rating]

    return recommendations.head(5)[['Title', 'Organization', 'Skills', 'Ratings', 'Difficulty', 'Type', 'Duration', 'course_url']]

# Streamlit UI
st.set_page_config(page_title="Course Recommendation System", page_icon="🎓", layout="wide")

# Custom CSS for Professional Styling - Shade of White for Background
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;  /* Light white shade */
        }
        .stTextInput>div>div>input {
            border: 2px solid #FFA500;
        }
        .stButton>button {
            background-color: #FFA500;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title('🎓 Course Recommendation System')
st.write('Find the best courses tailored to your needs.')

# Using tabs to organize content
tab1, tab2 = st.tabs(["🔍 Search", "📊 About"])

with tab1:
    st.sidebar.title("🎯 Course Filters")
    difficulty_filter = st.sidebar.selectbox("Difficulty Level", ["All", "Beginner", "Intermediate", "Advanced"])
    min_rating_filter = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 4.0)

    user_query = st.text_input('🔍 Search for courses', '')

    if user_query:
        difficulty = None if difficulty_filter == "All" else difficulty_filter
        with st.spinner("Fetching recommendations..."):
            recommendations = recommend_courses_qlm(user_query, difficulty, min_rating_filter)

        if not recommendations.empty:
            st.write("### Recommended Courses:")
            for index, row in recommendations.iterrows():
                st.subheader(row['Title'])
                st.markdown(f"**Organization**: {row['Organization']}")
                st.markdown(f"**Skills**: {row['Skills']}")
                st.markdown(f"**Rating**: {row['Ratings']} ⭐")
                st.markdown(f"**Difficulty**: {row['Difficulty']}")
                st.markdown(f"**Duration**: {row['Duration']} hours")
                st.markdown(f"[Course Link]({row['course_url']})")
                st.write("---")
        else:
            st.write("No recommendations found. Try a different query.")
    else:
        st.write("Please enter a query to get course recommendations.")

with tab2:
    st.write("### About This App")
    st.write("""
    This Course Recommendation System helps you find the best courses based on your interests, difficulty level, and ratings.
    Use the filters on the sidebar to refine your search, and enter a topic or skill in the search bar to get started.
    """)
    st.write("#### Features:")
    st.write("- **Easy to use**: Just type in a topic or skill.")
    st.write("- **Filters**: Customize your search with difficulty and rating filters.")
    st.write("- **Comprehensive**: Get detailed course information.")

