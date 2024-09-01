import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import csv

# Load the cleaned data
df = pd.read_csv('cleaned_courses.csv')

# Vectorize the combined features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Build the Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5, metric='cosine').fit(tfidf_matrix)

# Function to recommend courses based on user query
def recommend_courses(query, difficulty=None, min_rating=0):
    query_vector = vectorizer.transform([query])
    distances, indices = model.kneighbors(query_vector)
    recommendations = df.iloc[indices[0]]

    if difficulty:
        recommendations = recommendations[recommendations['Difficulty'] == difficulty]
    recommendations = recommendations[recommendations['Ratings'] >= min_rating]

    return recommendations[
        ['Title', 'Organization', 'Skills', 'Ratings', 'Difficulty', 'Type', 'Duration', 'course_url']]

# Streamlit UI
st.set_page_config(page_title="Course Recommendation System", page_icon="🎓", layout="wide")

# Custom CSS for Professional Styling
st.markdown(
    """
    <style>
    body {
        background-color: #FAEBD7;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #ADDFFF;
        border-radius: 10px;
        padding: 20px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2E4053;
        font-family: 'Helvetica Neue', sans-serif;
    }
    p, div, .markdown-text-container {
        color: #333333;
    }
    .stTextInput > div > div > input {
        border: 1px solid #b6c5e4;
        background-color: #f4faff;
        border-radius: 5px;
        padding: 8px;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #5DADE2;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #2874A6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
            recommendations = recommend_courses(user_query, difficulty, min_rating_filter)

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
    st.write("#### User Feedback")
    feedback = st.text_area("Leave your feedback here:")
    if st.button("Submit Feedback"):
        if feedback:
            with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([feedback])
            st.success("Thank you for your feedback!")
        else:
            st.error("Please enter your feedback before submitting.")
