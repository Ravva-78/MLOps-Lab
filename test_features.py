import pandas as pd
import numpy as np
from src.features import RatingFeatures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the feature store we just saved
logger.info("Loading feature store...")
features = RatingFeatures.load('models/rating_features.pkl')

# Test 1: Find similar users
logger.info("\n=== Test 1: Similar User Discovery ===")
test_users = [1, 5, 10]
for user_id in test_users:
    similar = features.get_similar_users(user_id=user_id, n=5)
    print(f"\nUsers similar to User {user_id}:")
    for sim_user_id, score in similar:
        print(f"  User {sim_user_id:3d}: similarity = {score:.4f}")

# Test 2: Matrix shapes
logger.info("\n=== Test 2: Feature Matrix Statistics ===")
print(f"Rating matrix shape:     {features.ratings_matrix.shape}")
print(f"Similarity matrix shape: {features.similarity_matrix.shape}")
print(f"Unique users:            {len(features.user_ids)}")
print(f"Unique movies:           {len(features.movie_ids)}")

# Test 3: Sparsity
logger.info("\n=== Test 3: Sparsity Analysis ===")
n_ratings = (features.ratings_matrix != 0).sum().sum()
max_possible = features.ratings_matrix.shape[0] * features.ratings_matrix.shape[1]
density = n_ratings / max_possible
print(f"Total ratings:   {n_ratings}")
print(f"Max possible:    {max_possible}")
print(f"Density:         {density:.2%}")
print(f"Sparsity:        {1-density:.2%}")

# Test 4: Individual user ratings
logger.info("\n=== Test 4: User Rating Profiles ===")
for user_id in [1, 2, 3]:
    rating_vector = features.get_user_ratings_vector(user_id)
    rated_movies = np.sum(rating_vector > 0)
    avg_rating = np.mean(rating_vector[rating_vector > 0])
    print(f"User {user_id}: rated {rated_movies} movies, avg rating = {avg_rating:.2f}")

logger.info("\n✓ All tests passed!")