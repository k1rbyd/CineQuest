import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle
import os
import logging
import ast  # For safely evaluating strings containing Python literals

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommendationModel:
    def __init__(self):
        logger.info("Initializing MovieRecommendationModel...")
        self.movies = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.svd = None
        self.load_data()

    def load_data(self):
        logger.info("Loading movie data...")
        try:
            # Check if preprocessed data exists
            if os.path.exists("preprocessed_data.pkl"):
                with open("preprocessed_data.pkl", "rb") as file:
                    self.movies, self.vectorizer, self.tfidf_matrix, self.svd = pickle.load(file)
                logger.info("Preprocessed data loaded successfully.")
            else:
                # Load movies metadata
                self.movies = pd.read_csv("movies_metadata.csv", low_memory=False)
                self.movies = self.movies[['id', 'title', 'overview', 'genres']]
                self.movies['overview'] = self.movies['overview'].fillna('')
                
                # Parse the 'genres' column (which contains JSON-like strings)
                self.movies['genres'] = self.movies['genres'].apply(
                    lambda x: " ".join([genre['name'] for genre in ast.literal_eval(x)]) if pd.notna(x) else ""
                )

                # Load keywords and merge with movies
                keywords = pd.read_csv("keywords.csv")
                keywords['keywords'] = keywords['keywords'].apply(
                    lambda x: " ".join([kw['name'] for kw in ast.literal_eval(x)]) if pd.notna(x) else ""
                )
                self.movies = self.movies.merge(keywords, on='id', how='left')
                self.movies['keywords'] = self.movies['keywords'].fillna('')

                # Load credits and merge with movies
                credits = pd.read_csv("credits.csv")
                credits['cast'] = credits['cast'].apply(
                    lambda x: " ".join([actor['name'] for actor in ast.literal_eval(x)][:5]) if pd.notna(x) else ""
                )
                credits['crew'] = credits['crew'].apply(
                    lambda x: " ".join([member['name'] for member in ast.literal_eval(x) if member['job'] == 'Director'][:2]) if pd.notna(x) else ""
                )
                self.movies = self.movies.merge(credits[['id', 'cast', 'crew']], on='id', how='left')
                self.movies['cast'] = self.movies['cast'].fillna('')
                self.movies['crew'] = self.movies['crew'].fillna('')

                # Combine all features for better recommendations
                self.movies['combined_features'] = (
                    self.movies['overview'] + " " +
                    self.movies['genres'] + " " +
                    self.movies['keywords'] + " " +
                    self.movies['cast'] + " " +
                    self.movies['crew']
                )

                # Compute TF-IDF
                self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  # Limit features for performance
                tfidf_matrix = self.vectorizer.fit_transform(self.movies['combined_features'])

                # Reduce dimensionality for faster computation
                self.svd = TruncatedSVD(n_components=100)
                self.tfidf_matrix = self.svd.fit_transform(tfidf_matrix)

                # Save preprocessed data for future use
                with open("preprocessed_data.pkl", "wb") as file:
                    pickle.dump((self.movies, self.vectorizer, self.tfidf_matrix, self.svd), file)
                logger.info("Movies metadata loaded and TF-IDF computed successfully.")
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")

    def get_recommendations(self, user_input):
        logger.info(f"Generating recommendations for: {user_input}")

        if not user_input:
            return ["No input provided. Please enter some preferences."]
        
        if self.tfidf_matrix is None or self.vectorizer is None:
            return ["Model not initialized. Please try again later."]
        
        # Convert user input into vector
        user_input_vector = self.vectorizer.transform([user_input])
        user_input_vector = self.svd.transform(user_input_vector)
        
        # Compute the cosine similarity between user input and movie descriptions
        cosine_similarities = cosine_similarity(user_input_vector, self.tfidf_matrix)
        
        # Get the top 10 most similar movies
        similar_movie_indices = cosine_similarities[0].argsort()[-10:][::-1]  # Get top 10
        
        recommendations = []
        seen_titles = set()  # To avoid recommending duplicate titles
        
        # Get movie titles for the top similar movies
        for index in similar_movie_indices:
            title = self.movies.iloc[index]['title']
            if title not in seen_titles:
                recommendations.append(title)
                seen_titles.add(title)
            if len(recommendations) == 5:  # Limit to top 5 recommendations
                break
        
        if not recommendations:
            return ["No recommendations found based on the input."]
        
        return recommendations

    def get_multiple_recommendations(self, user_inputs):
        all_recommendations = []
        for user_input in user_inputs:
            recommendations = self.get_recommendations(user_input)
            all_recommendations.extend(recommendations)
        
        # Remove duplicates
        recommendations = list(set(all_recommendations))
        return recommendations[:5]  # Limit to top 5 recommendations

# Train and save the model
if __name__ == "__main__":
    model = MovieRecommendationModel()  # Initialize and train the model
    
    # Save the trained model to a file
    try:
        with open("movie_recommendation_model.pkl", "wb") as file:
            pickle.dump(model, file)
        logger.info("✅ Model training completed and saved!")
    except Exception as e:
        logger.error(f"❌ Error saving model: {e}")