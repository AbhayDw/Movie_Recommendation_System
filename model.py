import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load pickle files
movies = pickle.load(open('df.pkl', 'rb'))

indices = pickle.load(open('indices.pkl', 'rb'))

tfidf = pickle.load(open('tfidf.pkl', 'rb'))

tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))

# Recommendation function
def recommend(movie_name):

    # Check if movie exists
    if movie_name not in indices:
        return None, None, None

    # Get movie index
    idx = indices[movie_name]

    # Calculate similarity dynamically for the requested movie only
    sim_scores_array = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get similarity scores
    sim_scores = list(enumerate(sim_scores_array))

    # Sort movies by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Remove first movie because it is same movie
    sim_scores = sim_scores[1:6]

    # Get recommended movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Get recommended titles
    recommended_titles = movies['title'].iloc[movie_indices].tolist()

    # Calculate similarity percentages
    similarity_percentages = [round(i[1] * 100, 1) for i in sim_scores]

    # Generate explanations based on shared genres
    target_genres = set(movies['genres'].iloc[idx].split())
    explanations = []
    for m_idx in movie_indices:
        rec_genres = set(movies['genres'].iloc[m_idx].split())
        shared = target_genres.intersection(rec_genres)
        if shared:
            explanations.append(f"Shares {', '.join(list(shared)[:2])} genres")
        else:
            explanations.append("Similar storyline and keywords")

    return recommended_titles, similarity_percentages, explanations