from flask import Flask, render_template, request
import pickle
import logging
import requests
import time
from main import MovieRecommendationModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# TMDb API setup
TMDB_API_KEY = '42a927985c260dc50bc9a9a1abf0fe4f'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500/'

# Load the trained model
try:
    with open("movie_recommendation_model.pkl", "rb") as file:
        model = pickle.load(file)
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None  # Prevent crashing

def get_movie_poster(movie_name, retries=3, delay=2):
    """
    Fetches movie poster URL from TMDb API.
    Retries a few times in case of a connection issue.
    """
    for attempt in range(retries):
        try:
            search_url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()
            
            if data['results']:
                movie = data['results'][0]
                poster_path = movie.get('poster_path')
                if poster_path:
                    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching poster for {movie_name}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                return None

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    posters = []
    error_message = None
    if request.method == "POST":
        user_preferences = request.form.get("preferences", "").strip()
        logger.info(f"📝 User input received: {user_preferences}")  # Debugging input

        if user_preferences and model:
            try:
                # Split the input by commas or spaces to get multiple preferences
                user_inputs = [input.strip() for input in user_preferences.split(",")]
                recommendations = model.get_multiple_recommendations(user_inputs)
                logger.info(f"🎬 Recommendations generated: {recommendations}")  # Debugging output

                # Fetch posters for each movie
                for movie in recommendations:
                    poster_url = get_movie_poster(movie)
                    posters.append(poster_url)
                
                logger.info(f"🎥 Movie posters fetched: {posters}")

            except Exception as e:
                logger.error(f"❌ Error in get_recommendations: {e}")
                error_message = "An error occurred while generating recommendations."
        else:
            error_message = "Please enter valid preferences."

    return render_template("index.html", recommendations=recommendations, posters=posters, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)