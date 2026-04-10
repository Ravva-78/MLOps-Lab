import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging
from typing import Tuple, List, Dict
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RatingFeatures:
    """
    Compute user-to-user similarity features from ratings matrix.
    
    This class:
    1. Converts ratings DataFrame → user × movie matrix
    2. Computes cosine similarity between all user pairs
    3. Enables fast similarity lookups for K-NN recommendations
    
    Example:
        ratings_df = pd.read_csv('data/ratings_clean.csv')
        features = RatingFeatures()
        features.fit(ratings_df)
        similar = features.get_similar_users(user_id=5, n=3)
        # Returns: [(42, 0.89), (17, 0.85), (103, 0.81)]
    """
    
    def __init__(self):
        """Initialize empty feature store."""
        self.ratings_matrix = None      # n_users × n_movies matrix
        self.similarity_matrix = None   # n_users × n_users similarity matrix
        self.user_ids = None            # Maps matrix index to user_id
        self.movie_ids = None           # Maps matrix col to movie_id
        self.fitted = False
    
    def fit(self, ratings_df: pd.DataFrame) -> 'RatingFeatures':
        """
        Learn rating features from training dataset.
        
        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating, timestamp]
                       Example:
                       user_id  movie_id  rating  timestamp
                       1        10        5.0     879024327
                       1        11        4.0     879024400
                       2        10        4.0     879024300
        
        Returns:
            self: Fitted RatingFeatures object
        
        Raises:
            ValueError: If required columns are missing or data is empty
            
        Process:
            1. Create user × movie pivot table (rating matrix)
            2. Fill missing ratings with 0.0 (unrated)
            3. Compute cosine similarity matrix
            4. Store user/movie ID mappings
            5. Log summary statistics
        """
        try:
            # Validate input
            if ratings_df.empty:
                raise ValueError("ratings_df cannot be empty")
            
            required_cols = {'user_id', 'movie_id', 'rating'}
            if not required_cols.issubset(ratings_df.columns):
                missing = required_cols - set(ratings_df.columns)
                raise ValueError(f"Missing required columns: {missing}")
            
            # Create user × movie rating matrix (sparse → dense)
            logger.info("Creating user × movie rating matrix...")
            self.ratings_matrix = ratings_df.pivot_table(
                index='user_id',
                columns='movie_id',
                values='rating',
                fill_value=0.0  # Unrated movies get 0.0
            )
            
            self.user_ids = self.ratings_matrix.index.values
            self.movie_ids = self.ratings_matrix.columns.values
            
            logger.info(f"  Shape: {self.ratings_matrix.shape[0]} users × "
                       f"{self.ratings_matrix.shape[1]} movies")
            
            # Compute sparsity
            n_ratings = (self.ratings_matrix != 0).sum().sum()
            max_possible = self.ratings_matrix.shape[0] * self.ratings_matrix.shape[1]
            sparsity = 1 - (n_ratings / max_possible)
            logger.info(f"  Density: {1-sparsity:.1%} (sparsity: {sparsity:.1%})")
            
            # Compute user-to-user cosine similarity
            logger.info("Computing cosine similarity matrix...")
            self.similarity_matrix = cosine_similarity(self.ratings_matrix)
            
            # Validate similarity matrix
            assert self.similarity_matrix.shape == (
                len(self.user_ids), len(self.user_ids)
            ), "Similarity matrix shape mismatch"
            
            # Set diagonal to 0 (don't recommend user to themselves)
            np.fill_diagonal(self.similarity_matrix, 0)
            
            self.fitted = True
            logger.info(f"✓ Feature engineering complete!")
            logger.info(f"  Fitted {len(self.user_ids)} users in similarity matrix")
            
            return self
        
        except Exception as e:
            logger.error(f"Failed to fit features: {str(e)}")
            raise
    
    def get_similar_users(
        self,
        user_id: int,
        n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find n most similar users to a target user.
        
        Args:
            user_id: Target user ID
            n: Number of similar users to return (default: 5)
        
        Returns:
            List of (user_id, similarity_score) tuples, sorted by
            similarity in descending order.
            Example: [(42, 0.89), (17, 0.85), (103, 0.81)]
        
        Raises:
            ValueError: If user_id not in training data
            RuntimeError: If model not fitted
            
        Algorithm:
            1. Find matrix index of user_id
            2. Extract similarity vector for that user
            3. Get top n indices (excluding self)
            4. Return user_ids and scores
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        try:
            # Convert user_id to matrix index
            user_indices = np.where(self.user_ids == user_id)[0]
            
            if len(user_indices) == 0:
                logger.warning(f"User {user_id} not in training data")
                return []
            
            user_idx = user_indices[0]
            
            # Get similarity scores for this user (already 0 on diagonal)
            similarities = self.similarity_matrix[user_idx]
            
            # Get top n similar users (already excludes self due to 0 diagonal)
            if np.sum(similarities > 0) == 0:
                logger.warning(f"User {user_id} has no similar users")
                return []
            
            # Get top n indices
            top_indices = np.argsort(similarities)[-n:][::-1]
            
            # Filter out zeros (no similarity)
            valid_indices = top_indices[similarities[top_indices] > 0]
            
            # Convert back to user IDs
            similar_user_ids = self.user_ids[valid_indices]
            similarity_scores = similarities[valid_indices]
            
            result = list(zip(similar_user_ids, similarity_scores))
            
            logger.debug(f"Found {len(result)} similar users for user {user_id}")
            return result
        
        except Exception as e:
            logger.error(f"Error finding similar users: {str(e)}")
            raise
    
    def get_user_ratings_vector(self, user_id: int) -> np.ndarray:
        """
        Get the rating vector for a specific user.
        
        Args:
            user_id: Target user ID
        
        Returns:
            Array of ratings for this user (0.0 for unrated movies)
        
        Raises:
            ValueError: If user_id not in training data
        """
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not in training data")
        
        user_idx = np.where(self.user_ids == user_id)[0][0]
        return self.ratings_matrix.iloc[user_idx].values
    
    def get_movie_rating_stats(self) -> Dict[str, float]:
        """
        Compute statistics on movie ratings in the training set.
        
        Returns:
            Dict with: mean_rating, std_rating, min_rating, max_rating
        """
        rated_values = self.ratings_matrix[self.ratings_matrix > 0].values.flatten()
        
        return {
            'mean_rating': float(np.mean(rated_values)),
            'std_rating': float(np.std(rated_values)),
            'min_rating': float(np.min(rated_values)),
            'max_rating': float(np.max(rated_values)),
            'n_rated': int(len(rated_values))
        }
    
    def save(self, filepath: str) -> None:
        """
        Serialize feature store to disk for production inference.
        
        Args:
            filepath: Path to save pickle file (typically 'models/rating_features.pkl')
        
        Note:
            Uses joblib for efficient pickle with numpy support.
            File can be loaded later with: features = RatingFeatures.load(filepath)
        """
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            joblib.dump(self, filepath)
            file_size_kb = os.path.getsize(filepath) / 1024
            logger.info(f"✓ Saved feature store: {filepath} ({file_size_kb:.1f} KB)")
        except Exception as e:
            logger.error(f"Failed to save features: {str(e)}")
            raise
    
    @staticmethod
    def load(filepath: str) -> 'RatingFeatures':
        """
        Load serialized feature store from disk.
        
        Args:
            filepath: Path to pickle file
        
        Returns:
            RatingFeatures object (already fitted)
        
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If deserialization fails
        """
        try:
            features = joblib.load(filepath)
            if not isinstance(features, RatingFeatures):
                raise TypeError("Loaded object is not RatingFeatures")
            logger.info(f"✓ Loaded feature store: {filepath}")
            return features
        except FileNotFoundError:
            logger.error(f"Feature file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Failed to load features: {str(e)}")
            raise