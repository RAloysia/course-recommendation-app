import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# Load the cleaned data from the GitHub repository
csv_url = "https://raw.githubusercontent.com/RAloysia/course-recommendation-app/main/cleaned_courses.csv"
response = requests.get(csv_url)
response.raise_for_status()  # Ensure the request was successful

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(StringIO(response.text))

# Ensure the combined features column exists and is filled correctly
if 'combined_features' not in df.columns:
    st.error("Error: 'combined_features' column is missing in the dataset.")
else:
    df['combined_features'] = df['combined_features'].fillna('').str.lower()

# Handle missing or invalid URLs by replacing NaN with an empty string
df['course_url'] = df['course_url'].fillna('https://www.coursera.org/?skipBrowseRedirect=true')

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

# Load background image using PIL
bg_image = Image.open("learning_bg.jpeg")  # Adjust the path accordingly

# Display background image as full-screen
st.image(bg_image, use_column_width=True)

st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #c0ebe8;  /* Set the background color to the desired value */
    }
    .sidebar .sidebar-content {
        background-color: rgba(192, 235, 232, 0.8);  /* Semi-transparent sidebar with RGB color */
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }
    .sidebar .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: #010345;
    }
    .sidebar .sidebar-filter {
        margin-top: 10px;
        padding: 10px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-image {
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }
    .stTab {
        background-color: rgba(210, 250, 243);  /* Semi-transparent tabs */
    }
    
    .title {
        position: absolute;
        top: -250px;
        left: 30px;
        color: #010345;
        font-size: 5vw; /* Responsive font size based on viewport width */
        font-weight: bold;
        z-index: 10;
    }

    /* Responsive adjustments for smaller screens */
    @media (max-width: 768px) {
        .title {
            font-size: 6vw; /* Slightly larger on smaller screens */
            top: -150px; /* Adjust position for mobile */
            left: 15px;  /* Adjust left position */
        }
    }

    @media (max-width: 480px) {
        .title {
            font-size: 8vw; /* Even larger on very small screens */
            top: -120px;
            left: 10px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)
# App title and description overlay
st.markdown('<div class="title">🎓 Course Recommendation System</div>', unsafe_allow_html=True)

# Using tabs to organize content
tab1, tab2 = st.tabs(["🔍 Search", "📊 About"])

with tab1:
    # Display the sidebar image
    # Sidebar image
    st.sidebar.image("online.gif", use_column_width=True)

    st.sidebar.markdown('<div class="sidebar-title">🎯 Course Filters</div>', unsafe_allow_html=True)
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
                st.markdown(f"*Organization*: {row['Organization']}")
                st.markdown(f"*Skills*: {row['Skills']}")
                st.markdown(f"*Rating*: {row['Ratings']} ⭐")
                st.markdown(f"*Difficulty*: {row['Difficulty']}")
                st.markdown(f"*Duration*: {row['Duration']} hours")

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
    st.write("""This Course Recommendation System helps you find the best courses based on your interests, difficulty level, and ratings.
    Use the filters on the sidebar to refine your search, and enter a topic or skill in the search bar to get started.
    """)
    st.write("#### Features:")
    st.write("- *Easy to use*: Just type in a topic or skill.")
    st.write("- *Filters*: Customize your search with difficulty and rating filters.")
    st.write("- *Comprehensive*: Get detailed course information.")
