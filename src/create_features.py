import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_user_similarity():
    """Build user-user similarity matrix from ratings."""
    logger.info("Loading clean ratings...")
    df = pd.read_csv('data/processed/ratings_clean.csv')
    
    # Create user × movie matrix
    # Rows: users, Columns: movies, Values: ratings (0 if not rated)
    ratings_matrix = df.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating',
        fill_value=0.0
    )
    
    logger.info(f"Ratings matrix shape: {ratings_matrix.shape}")
    
    # Compute cosine similarity between users
    # cosine_similarity = 1 means identical rating patterns
    # cosine_similarity = 0 means no correlation
    similarity_matrix = cosine_similarity(ratings_matrix)
    
    # Save features
    features = {
        'similarity': similarity_matrix,
        'user_ids': ratings_matrix.index.values,
        'movie_ids': ratings_matrix.columns.values,
        'ratings_df': df
    }
    
    joblib.dump(features, 'models/user_similarity.pkl')
    
    logger.info(f"✓ Created {similarity_matrix.shape} similarity matrix")
    logger.info(f"Saved to: models/user_similarity.pkl")

if __name__ == '__main__':
    create_user_similarity()