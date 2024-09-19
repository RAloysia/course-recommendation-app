import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load the cleaned data from the GitHub repository
csv_url = "https://raw.githubusercontent.com/RAloysia/course-recommendation-app/main/cleaned_courses.csv"
response = requests.get(csv_url)
response.raise_for_status()  # Ensure the request was successful

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(StringIO(response.text))

# Preprocess the combined features
df['combined_features'] = df['combined_features'].fillna('').str.lower()

# Handle missing or invalid URLs by replacing NaN with an empty string
df['course_url'] = df['course_url'].fillna('')

# Vectorize the combined features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Build the Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5, metric='cosine').fit(tfidf_matrix)


# Function to recommend courses based on user query with filters
def recommend_courses(query, difficulty=None, min_rating=0):
    query_vector = vectorizer.transform([query])
    distances, indices = model.kneighbors(query_vector)
    recommendations = df.iloc[indices[0]][
        ['Title', 'Organization', 'Skills', 'Ratings', 'Difficulty', 'Type', 'Duration', 'course_url']]

    # Apply filters: Difficulty and Minimum Rating
    if difficulty and difficulty != 'All':
        recommendations = recommendations[recommendations['Difficulty'] == difficulty]
    recommendations = recommendations[recommendations['Ratings'] >= min_rating]

    return recommendations


# Streamlit app UI
st.set_page_config(page_title="Course Recommendation System", page_icon="🎓", layout="wide")

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
            recommended_courses = recommend_courses(user_query, difficulty, min_rating_filter)

        if not recommended_courses.empty:
            st.write("### Recommended Courses:")
            for index, row in recommended_courses.iterrows():
                st.subheader(row['Title'])
                st.markdown(f"**Organization**: {row['Organization']}")
                st.markdown(f"**Skills**: {row['Skills']}")
                st.markdown(f"**Rating**: {row['Ratings']} ⭐")
                st.markdown(f"**Difficulty**: {row['Difficulty']}")
                st.markdown(f"**Duration**: {row['Duration']} hours")

                # Check if the course_url exists and is valid
                if pd.notna(row['course_url']) and row['course_url'].startswith("http"):
                    st.markdown(f"[Course Link]({row['course_url']})")
                else:
                    st.markdown("Course link not available.")

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
