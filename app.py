import streamlit as st
import pickle
import requests
import os
from dotenv import load_dotenv
from model import recommend
import random
import pandas as pd

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API")
if OMDB_API_KEY: OMDB_API_KEY = OMDB_API_KEY.strip()

HF_API_KEY = os.getenv("hugging_face_api")
if HF_API_KEY: HF_API_KEY = HF_API_KEY.strip()

# Page Configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Watchlist
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

def add_to_watchlist(movie_title):
    if movie_title not in st.session_state.watchlist:
        st.session_state.watchlist.append(movie_title)
        st.toast(f"Added {movie_title} to Watchlist!")
    else:
        st.toast(f"{movie_title} is already in your Watchlist.")

def remove_from_watchlist(movie_title):
    if movie_title in st.session_state.watchlist:
        st.session_state.watchlist.remove(movie_title)
        st.toast(f"Removed {movie_title} from Watchlist!")

# Fetch movie details from OMDb API
@st.cache_data(ttl=3600)
def fetch_movie_details(movie_name):
    if not OMDB_API_KEY: return None
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("Response") == "True":
            return {
                "poster": data.get("Poster", "N/A"),
                "rating": data.get("imdbRating", "N/A"),
                "year": data.get("Year", "N/A"),
                "plot": data.get("Plot", "No plot available.")
            }
        else:
            return None # API error or movie not found in OMDb
    except requests.exceptions.RequestException as e:
        # Handle internet issues or API failure gracefully
        return "network_error"
    except Exception as e:
        return None

def display_movie_card(title, details, unique_key, similarity=None, explanation=None):
    if details == "network_error":
        st.error("Network error: Could not load details.")
        st.image("https://via.placeholder.com/300x450?text=No+Internet", use_container_width=True)
        year, rating, plot = 'N/A', 'N/A', 'No plot available due to network error.'
    else:
        if details and details.get('poster') and details['poster'] != "N/A":
            st.image(details['poster'], use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
        
        year = details['year'] if details else 'N/A'
        rating = details['rating'] if details else 'N/A'
        plot = details['plot'] if details else 'No plot available.'
    
    st.markdown(f"**{title}** ({year})")
    st.markdown(f"⭐ **{rating}**/10")
    
    if similarity is not None:
        st.markdown(f"🎯 **Similarity Score: {similarity}%**")
        
    if explanation:
        with st.expander("Why recommended?"):
            st.markdown("Recommended because:")
            st.markdown(f"- {explanation}")
            st.markdown("- Similar storyline")
            st.markdown("- Similar keywords")
    
    # Trailer Link
    yt_query = title.replace(' ', '+') + "+official+trailer"
    st.link_button("▶️ Watch Trailer", f"https://www.youtube.com/results?search_query={yt_query}")
    
    if st.button("➕ Watchlist", key=f"add_{unique_key}_{title}"):
        add_to_watchlist(title)
        
    with st.expander("View Plot"):
        st.write(plot)

# Emotion Detection via Hugging Face
def detect_emotion(text):
    if not HF_API_KEY:
        return "neutral", "HF API Key not found"
    
    API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            emotions = data[0]
            top_emotion = max(emotions, key=lambda x: x['score'])
            return top_emotion['label'], None
        elif isinstance(data, dict) and "error" in data:
            return "neutral", data["error"]
    except Exception as e:
        return "neutral", str(e)
    return "neutral", "Unknown error"

EMOTION_GENRE_MAP = {
    "joy": ["Comedy", "Animation", "Family"],
    "sadness": ["Comedy", "Romance", "Adventure"],
    "anger": ["Action", "Thriller", "Crime"],
    "fear": ["Horror", "Mystery", "Thriller"],
    "surprise": ["Sci-Fi", "Fantasy", "Mystery"],
    "disgust": ["Documentary", "Drama"],
    "neutral": ["Drama", "Romance", "History"]
}

# Load movie dataset
@st.cache_data
def load_data():
    return pickle.load(open('df.pkl', 'rb'))

try:
    movies_df = load_data()
    movie_titles = movies_df['title'].values
except Exception as e:
    st.error(f"Error loading movie data: {e}")
    movie_titles = []
    movies_df = None

# UI Rendering
st.title("🎬 Movie Recommendation System")
st.markdown("Discover your next favorite movie based on what you already love or how you feel!")

tab1, tab2, tab3, tab4 = st.tabs(["🔥 Trending", "🔍 Search by Movie", "🧠 Search by Mood", "📋 My Watchlist"])

with tab1:
    st.header("🔥 Trending Now")
    if movies_df is not None and 'popularity' in movies_df.columns:
        # Safely convert to numeric before sorting to avoid TypeError
        pop_series = pd.to_numeric(movies_df['popularity'], errors='coerce')
        trending_indices = pop_series.sort_values(ascending=False).head(5).index
        trending_movies = movies_df.loc[trending_indices, 'title'].tolist()
        cols = st.columns(5)
        for i, title in enumerate(trending_movies):
            with cols[i]:
                details = fetch_movie_details(title)
                display_movie_card(title, details, "trend")

with tab2:
    st.header("Find Similar Movies")
    selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_titles)

    if st.button("Recommend Movies"):
        with st.spinner("Finding best movies for you..."):
            recommendations, similarities, explanations = recommend(selected_movie)
            
            if not recommendations:
                st.error(f"Error: We could not find '{selected_movie}' in our database. Please select a valid movie.")
            else:
                st.subheader(f"Because you liked '{selected_movie}':")
                cols = st.columns(5)
                for i, title in enumerate(recommendations):
                    with cols[i]:
                        details = fetch_movie_details(title)
                        display_movie_card(title, details, f"rec_{i}", similarity=similarities[i], explanation=explanations[i])

with tab3:
    st.header("Emotion-Based Recommendations")
    user_mood = st.text_input("How are you feeling right now? (e.g., 'I am feeling very happy today!', 'I had a stressful day')")
    
    if st.button("Detect Mood & Recommend"):
        if user_mood:
            with st.spinner("Analyzing your mood..."):
                emotion, error = detect_emotion(user_mood)
                
                if error and "Model" not in error:
                    st.warning(f"Note: Using default 'neutral' mood due to API issue ({error})")
                
                st.success(f"Detected Mood: **{emotion.capitalize()}**")
                genres = EMOTION_GENRE_MAP.get(emotion, ["Drama"])
                st.info(f"Based on your mood, recommending genres: {', '.join(genres)}")
                
                if movies_df is not None:
                    matched_movies = movies_df[movies_df['genres'].str.contains('|'.join(genres), case=False, na=False)]
                    if not matched_movies.empty:
                        sample_size = min(5, len(matched_movies))
                        recommended_mood_movies = matched_movies.sample(sample_size)['title'].tolist()
                        
                        cols = st.columns(5)
                        for i, title in enumerate(recommended_mood_movies):
                            with cols[i]:
                                details = fetch_movie_details(title)
                                display_movie_card(title, details, f"mood_{i}")
                    else:
                        st.warning("Could not find matching movies for this mood.")
        else:
            st.warning("Please enter your mood first.")

with tab4:
    st.header("📋 My Watchlist")
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add some movies!")
    else:
        cols = st.columns(5)
        for i, title in enumerate(st.session_state.watchlist):
            with cols[i % 5]:
                details = fetch_movie_details(title)
                if details and details['poster'] != "N/A":
                    st.image(details['poster'], use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                st.markdown(f"**{title}**")
                if st.button("❌ Remove", key=f"rem_watch_{title}"):
                    remove_from_watchlist(title)
                    st.rerun()
