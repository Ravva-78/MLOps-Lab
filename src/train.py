import argparse
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Tuple, Optional
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KNNRecommendationModel:
    """
    K-Nearest Neighbors recommendation model.
    
    Predicts ratings using collaborative filtering:
    1. Find K users most similar to target user
    2. Average their ratings for the target movie
    3. Return weighted or unweighted average
    
    Example:
        model = KNNRecommendationModel(k=5)
        model.fit(features, ratings_df)
        prediction = model.predict_rating(user_id=42, movie_id=7)
        # Returns: 4.5 (predicted rating)
    """
    
    def __init__(self, k: int = 5, use_similarity_weights: bool = False):
        """
        Initialize K-NN model.
        
        Args:
            k: Number of nearest neighbors to use for prediction
            use_similarity_weights: If True, weight neighbors by similarity.
                                   If False, all neighbors weighted equally.
        
        Example:
            model1 = KNNRecommendationModel(k=5)  # Simple average
            model2 = KNNRecommendationModel(k=5, use_similarity_weights=True)
                                                    # Weighted average
        """
        self.k = k
        self.use_similarity_weights = use_similarity_weights
        self.features = None           # Lab 5 RatingFeatures object
        self.ratings_df = None         # Training ratings data
        self.default_rating = 3.0      # Default if no neighbors found
        self.fitted = False
        
        logger.debug(f"Initialized KNN(k={k}, weighted={use_similarity_weights})")
    
    def fit(
        self,
        features,
        ratings_df: pd.DataFrame
    ) -> 'KNNRecommendationModel':
        """
        Store feature extractor and training ratings.
        
        Note: K-NN doesn't actually "learn" in traditional sense.
        We just store references to the similarity matrix and ratings.
        
        Args:
            features: Fitted RatingFeatures object from Lab 5
            ratings_df: Training ratings DataFrame
        
        Returns:
            self: Fitted model object
        
        Raises:
            ValueError: If features not fitted or ratings_df invalid
        """
        try:
            if not hasattr(features, 'fitted') or not features.fitted:
                raise ValueError("features must be fitted RatingFeatures object")
            
            if ratings_df.empty:
                raise ValueError("ratings_df cannot be empty")
            
            required_cols = {'user_id', 'movie_id', 'rating'}
            if not required_cols.issubset(ratings_df.columns):
                missing = required_cols - set(ratings_df.columns)
                raise ValueError(f"Missing columns: {missing}")
            
            self.features = features
            self.ratings_df = ratings_df
            self.fitted = True
            
            logger.info(f"✓ Fitted KNN model")
            logger.info(f"  K: {self.k}")
            logger.info(f"  Similarity weights: {self.use_similarity_weights}")
            logger.info(f"  Training samples: {len(ratings_df)}")
            
            return self
        
        except Exception as e:
            logger.error(f"Failed to fit model: {str(e)}")
            raise
    
    def predict_rating(
        self,
        user_id: int,
        movie_id: int
    ) -> float:
        """
        Predict a single user's rating for a movie using K-NN.
        
        Algorithm:
            1. Find K users most similar to target user
            2. Filter to those who rated the target movie
            3. Average their ratings (optionally weighted by similarity)
            4. Return average or default if no neighbors found
        
        Args:
            user_id: Target user
            movie_id: Target movie
        
        Returns:
            Predicted rating (0.5 to 5.0) or default_rating if N/A
        
        Args:
            user_id: Target user ID
            movie_id: Target movie ID
        
        Returns:
            Float prediction in range [0.5, 5.0]
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        try:
            # Get K similar users
            similar_users = self.features.get_similar_users(
                user_id, 
                n=self.k
            )
            
            if not similar_users:
                logger.debug(f"No similar users for user {user_id}")
                return self.default_rating
            
            # Extract user IDs and similarities
            similar_user_ids = [u for u, s in similar_users]
            similarities = np.array([s for u, s in similar_users])
            
            # Find ratings from similar users for this movie
            ratings_for_movie = self.ratings_df[
                (self.ratings_df['user_id'].isin(similar_user_ids))
                & (self.ratings_df['movie_id'] == movie_id)
            ]
            
            # No neighbors rated this movie
            if ratings_for_movie.empty:
                logger.debug(
                    f"No similar user rated movie {movie_id} for user {user_id}"
                )
                return self.default_rating
            
            # Get ratings in same order as similar_users
            ratings_list = []
            similarity_list = []
            for neighbor_id, sim_score in similar_users:
                neighbor_rating = ratings_for_movie[
                    ratings_for_movie['user_id'] == neighbor_id
                ]['rating'].values
                
                if len(neighbor_rating) > 0:
                    ratings_list.append(neighbor_rating[0])
                    similarity_list.append(sim_score)
            
            if len(ratings_list) == 0:
                return self.default_rating
            
            # Average: weighted or unweighted
            if self.use_similarity_weights:
                similarity_list = np.array(similarity_list)
                weights = similarity_list / np.sum(similarity_list)
                prediction = np.sum(np.array(ratings_list) * weights)
            else:
                prediction = np.mean(ratings_list)
            
            # Clip to rating range
            prediction = np.clip(prediction, 0.5, 5.0)
            
            return float(prediction)
        
        except Exception as e:
            logger.error(f"Prediction error for user {user_id}, movie {movie_id}: {e}")
            return self.default_rating
    
    def predict_batch(
        self,
        test_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict ratings for multiple (user, movie) pairs.
        
        Args:
            test_df: DataFrame with columns [user_id, movie_id, ...]
                    Prediction ignores actual ratings in this df
        
        Returns:
            Array of predictions, same length as test_df
        
        Performance:
            ~1000 predictions/second on modern hardware
            Use batch for fast inference
        """
        predictions = np.zeros(len(test_df))
        
        for i, (idx, row) in enumerate(test_df.iterrows()):
            predictions[i] = self.predict_rating(
                int(row['user_id']),
                int(row['movie_id'])
            )
            
            # Log progress every 100 predictions
            if (i + 1) % 100 == 0:
                logger.debug(f"  Predicted {i + 1}/{len(test_df)} ratings")
        
        return predictions
    
    def get_config(self) -> dict:
        """Return model configuration for logging/saving."""
        return {
            'model_type': 'KNNRecommendation',
            'k': self.k,
            'use_similarity_weights': self.use_similarity_weights,
            'default_rating': self.default_rating
        }
    
    def save(self, filepath: str) -> None:
        """Save model to disk for production inference."""
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            joblib.dump(self, filepath)
            logger.info(f"✓ Saved model: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    @staticmethod
    def load(filepath: str) -> 'KNNRecommendationModel':
        """Load model from disk."""
        try:
            model = joblib.load(filepath)
            logger.info(f"✓ Loaded model: {filepath}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise