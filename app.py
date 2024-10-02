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
bg_image = Image.open("images/learning_bg.jpeg")  # Adjust the path accordingly

# Display background image as full-screen
st.image(bg_image, use_column_width=True)

st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #c0ebe8;
    }
    .sidebar .sidebar-content {
        background-color: rgba(192, 235, 232, 0.8);
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }
    .title {
        position: absolute;
        top: -250px;
        left: 35px;
        color: #010345;
        font-size: 3.5vw; 
        font-weight: bold;
        z-index: 10;
    }
    @media (max-width: 780px) {
        .title {
            font-size: 6vw; 
            top: -150px; 
            left: 15px;
        }
    }
    @media (max-width: 480px) {
        .title {
            font-size: 5.5vw;
            top: -160px;
            left: 10px;
        }
    }
    .course-container {
        display: block;
        justify-content: center;
        width: 100%;
        padding: 0 10px; /* Add padding for spacing */
    }
    .course-box {
        width: 100%;  /* Full page width */
        border: 1px solid #d3d3d3;
        padding: 20px;
        border-radius: 8px;
        background-color: #f0f8ff;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;  /* Vertical spacing */
        transition: all 0.3s ease;
    }
    .course-box:hover {
        transform: scale(1.05);
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.2);
    }
    .course-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .course-link {
        font-size: 14px;
        color: #0056b3;
        text-decoration: none;
        font-weight: bold;
    }
    .course-link:hover {
        text-decoration: underline;
    }
        /* Responsive design */
    @media (max-width: 1200px) {
        .course-box {
            padding: 15px;
        }
        .course-title {
            font-size: 16px;
        }
    }
    
    @media (max-width: 768px) {
        .course-box {
            padding: 10px;
        }
        .course-title {
            font-size: 14px;
        }
    }

    @media (max-width: 480px) {
        .course-box {
            padding: 8px;
        }
        .course-title {
            font-size: 12px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)



# App title and description overlay
st.markdown('<div class="title">🎓 Course Recommendation System</div>', unsafe_allow_html=True)

# Using tabs to organize content
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 About", "📚 All Courses"])

with tab1:
    st.sidebar.image("images/online.gif", use_column_width=True)

    st.sidebar.markdown('<div class="sidebar-title">🎯 Course Filters</div>', unsafe_allow_html=True)
    difficulty_filter = st.sidebar.selectbox("Difficulty Level", ["All", "Beginner", "Intermediate", "Advanced"])
    min_rating_filter = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 4.0)

    user_query = st.text_input('🔍 Search for courses', '')

    if user_query:
        difficulty = None if difficulty_filter == "All" else difficulty_filter
        with st.spinner("Fetching recommendations..."):
            recommended_courses = recommend_courses(user_query, difficulty, min_rating_filter)

        if not recommended_courses.empty:
            st.markdown('<div class="course-container">', unsafe_allow_html=True)  # Start the container

            for index, row in recommended_courses.iterrows():
                st.markdown(
                    f"""
                    <div class="course-box">
                        <div class="course-title">
                            {row['Title']}
                        </div>
                        <div class="course-details">
                            <p><strong>Organization:</strong> {row['Organization']}</p>
                            <p><strong>Skills:</strong> {row['Skills']}</p>
                            <p><strong>Rating:</strong> {row['Ratings']} ⭐</p>
                            <p><strong>Difficulty:</strong> {row['Difficulty']}</p>
                            <p><strong>Duration:</strong> {row['Duration']} hours</p>
                        </div>
                        <a class="course-link" href="{row['course_url']}" target="_blank">
                            Course Link
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)  # End the container
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

with tab3:
    st.write("### All Available Courses")
    st.write(f"Total number of courses: {len(df)}")

    # Define the number of columns (e.g., 3)
    num_columns = 3
    columns = st.columns(num_columns)

    # Iterate over the DataFrame and display the courses in columns
    for index, row in df.iterrows():
        column_index = index % num_columns
        with columns[column_index]:
            st.markdown(
                f"""
                <div class="course-box">
                    <div class="course-title">{row['Title']}</div>
                    <div class="course-details">
                        <p><strong>Rating:</strong> {row['Ratings']} ⭐</p>
                        <p><strong>Difficulty:</strong> {row['Difficulty']}</p>
                        <p><strong>Duration:</strong> {row['Duration']} hours</p>
                    </div>
                    <a class="course-link" href="{row['course_url']}" target="_blank">Course Link</a>
                </div>
                """,
                unsafe_allow_html=True
            )
