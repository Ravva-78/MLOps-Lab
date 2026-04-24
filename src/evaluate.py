import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load trained recommendation model."""
    try:
        model = joblib.load(model_path)
        logger.info(f"✓ Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_metadata(metadata_path: str) -> Dict:
    """Load model metadata (hyperparameters, training config)."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"✓ Loaded metadata from {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise


# ============================================================================
# RATING PREDICTION METRICS
# ============================================================================

def evaluate_rating_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """
    Compute error metrics for rating predictions.
    
    Args:
        y_true: Actual ratings from test set
        y_pred: Model predictions
    
    Returns:
        Dict with RMSE, MAE, count, and additional statistics
    
    Metrics:
        - RMSE (Root Mean Square Error): Best for penalizing outliers
        - MAE (Mean Absolute Error): Best for interpretability
        - Median error: Robust to extreme outliers
        - Percentiles: Shows error distribution
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    # Compute basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Compute error for deeper analysis
    errors = np.abs(y_true - y_pred)
    
    # Percentiles
    percentiles = np.percentile(errors, [25, 50, 75, 90, 95])
    
    logger.info(f"\n{'=' * 70}")
    logger.info("RATING PREDICTION EVALUATION")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total samples: {len(y_true)}")
    logger.info(f"\nPointwise Metrics:")
    logger.info(f"  RMSE:   {rmse:.4f}")
    logger.info(f"  MAE:    {mae:.4f}")
    logger.info(f"  Median: {np.median(errors):.4f}")
    logger.info(f"  Mean:   {errors.mean():.4f}")
    logger.info(f"  Std:    {errors.std():.4f}")
    logger.info(f"\nError Percentiles:")
    logger.info(f"  25th:   {percentiles[0]:.4f}")
    logger.info(f"  50th:   {percentiles[1]:.4f}")
    logger.info(f"  75th:   {percentiles[2]:.4f}")
    logger.info(f"  90th:   {percentiles[3]:.4f}")
    logger.info(f"  95th:   {percentiles[4]:.4f}")
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'median_error': float(np.median(errors)),
        'mean_error': float(errors.mean()),
        'std_error': float(errors.std()),
        'min_error': float(errors.min()),
        'max_error': float(errors.max()),
        'percentiles': {
            'p25': float(percentiles[0]),
            'p50': float(percentiles[1]),
            'p75': float(percentiles[2]),
            'p90': float(percentiles[3]),
            'p95': float(percentiles[4])
        },
        'n_samples': int(len(y_true))
    }


# ============================================================================
# COVERAGE ANALYSIS
# ============================================================================

def compute_coverage(
    model,
    test_df: pd.DataFrame,
    min_prediction: float = 0.5
) -> Dict:
    """
    Compute catalog coverage: % of movies the model recommends.
    
    Algorithm:
        For each movie in test set:
            For each user who rated it:
                Get model prediction
                If prediction >= min_prediction: mark as "recommended"
        Coverage = # unique recommended movies / total unique movies
    
    Args:
        model: Trained recommendation model
        test_df: Test ratings DataFrame
        min_prediction: Threshold for "recommending" a movie
    
    Returns:
        Dict with coverage %, recommended count, total catalogue size
    
    Interpretation:
        - High coverage (>80%): Model uses full catalogue
        - Low coverage (<50%): Model only recommends obvious movies
        - Baseline comparison: Random always has 100% coverage
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("CATALOG COVERAGE ANALYSIS")
    logger.info(f"{'=' * 70}")
    
    all_movies = test_df['movie_id'].unique()
    recommended_movies = set()
    
    # Check which movies get recommended
    for movie_id in all_movies:
        movie_ratings = test_df[test_df['movie_id'] == movie_id]
        
        for user_id in movie_ratings['user_id'].unique():
            pred_rating = model.predict_rating(
                int(user_id),
                int(movie_id)
            )
            
            # If model predicts >= threshold, it "recommends" this movie
            if pred_rating >= min_prediction:
                recommended_movies.add(int(movie_id))
                break  # One recommendation is enough to count as "covered"
    
    coverage = len(recommended_movies) / len(all_movies)
    
    logger.info(f"Total movies in catalog: {len(all_movies)}")
    logger.info(f"Movies recommended:      {len(recommended_movies)}")
    logger.info(f"Coverage ratio:          {coverage:.2%}")
    
    # Interpret coverage
    if coverage > 0.8:
        logger.info(f"✓ Good coverage: model uses most of catalog")
    elif coverage > 0.5:
        logger.info(f"⚠ Moderate coverage: model misses some movies")
    else:
        logger.info(f"✗ Low coverage: model too conservative")
    
    return {
        'coverage_ratio': float(coverage),
        'n_recommended': int(len(recommended_movies)),
        'n_total': int(len(all_movies)),
        'recommendation_threshold': min_prediction
    }


# ============================================================================
# SPARSITY & DATA STATISTICS
# ============================================================================

def analyze_sparsity(
    ratings_df: pd.DataFrame,
    n_movies: int
) -> Dict:
    """
    Analyze data sparsity: how full is the rating matrix?
    
    Args:
        ratings_df: All ratings (train + test combined)
        n_movies: Total number of movies in catalog
    
    Returns:
        Dict with density %, sparsity %, and sample counts
    
    Interpretation:
        - Density 10%: Only 1 in 10 possible (user, movie) pairs have ratings
        - This explains why K-NN needs similar users (can't rate everything)
        - Lower density → harder prediction problem
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("DATA SPARSITY ANALYSIS")
    logger.info(f"{'=' * 70}")
    
    n_users = ratings_df['user_id'].nunique()
    n_ratings = len(ratings_df)
    max_possible = n_users * n_movies
    
    density = n_ratings / max_possible
    sparsity = 1 - density
    
    logger.info(f"Users:           {n_users}")
    logger.info(f"Movies:          {n_movies}")
    logger.info(f"Max possible:    {max_possible:,}")
    logger.info(f"Actual ratings:  {n_ratings:,}")
    logger.info(f"Density:         {density:.2%}")
    logger.info(f"Sparsity:        {sparsity:.2%}")
    
    # User-level statistics
    ratings_per_user = ratings_df.groupby('user_id').size()
    logger.info(f"\nRatings per user:")
    logger.info(f"  Mean:   {ratings_per_user.mean():.1f}")
    logger.info(f"  Median: {ratings_per_user.median():.1f}")
    logger.info(f"  Min:    {ratings_per_user.min()}")
    logger.info(f"  Max:    {ratings_per_user.max()}")
    
    # Movie-level statistics
    ratings_per_movie = ratings_df.groupby('movie_id').size()
    logger.info(f"\nRatings per movie:")
    logger.info(f"  Mean:   {ratings_per_movie.mean():.1f}")
    logger.info(f"  Median: {ratings_per_movie.median():.1f}")
    logger.info(f"  Min:    {ratings_per_movie.min()}")
    logger.info(f"  Max:    {ratings_per_movie.max()}")
    
    return {
        'n_users': int(n_users),
        'n_movies': int(n_movies),
        'n_ratings': int(n_ratings),
        'max_possible': int(max_possible),
        'density': float(density),
        'sparsity': float(sparsity),
        'ratings_per_user': {
            'mean': float(ratings_per_user.mean()),
            'median': float(ratings_per_user.median()),
            'min': int(ratings_per_user.min()),
            'max': int(ratings_per_user.max())
        },
        'ratings_per_movie': {
            'mean': float(ratings_per_movie.mean()),
            'median': float(ratings_per_movie.median()),
            'min': int(ratings_per_movie.min()),
            'max': int(ratings_per_movie.max())
        }
    }


# ============================================================================
# ERROR DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """
    Analyze how errors are distributed.
    
    Why it matters:
        A model with MAE=0.6 is great if all errors are 0.6 ± 0.1
        But terrible if 90% of errors are 0.1 and 10% are 5.0+
    
    Args:
        y_true: Actual ratings
        y_pred: Predictions
    
    Returns:
        Dict with error distribution breakdown
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("ERROR DISTRIBUTION ANALYSIS")
    logger.info(f"{'=' * 70}")
    
    errors = np.abs(y_true - y_pred)
    
    logger.info(f"Prediction Range:    [{y_pred.min():.1f}, {y_pred.max():.1f}]")
    logger.info(f"Actual Range:        [{y_true.min():.1f}, {y_true.max():.1f}]")
    logger.info(f"Error Statistics:")
    logger.info(f"  Mean:   {errors.mean():.4f}")
    logger.info(f"  Std:    {errors.std():.4f}")
    logger.info(f"  Min:    {errors.min():.4f}")
    logger.info(f"  Max:    {errors.max():.4f}")
    
    # Count errors by range (for rating scale [0.5, 5.0])
    ranges = [
        (0.0, 0.5),   # Perfect to near-perfect
        (0.5, 1.0),   # Good
        (1.0, 1.5),   # Acceptable
        (1.5, 2.0),   # Poor
        (2.0, 5.0)    # Very poor
    ]
    
    logger.info(f"\nError Distribution by Range:")
    distribution = {}
    
    for low, high in ranges:
        count = np.sum((errors >= low) & (errors < high))
        pct = 100 * count / len(errors)
        bar = "█" * int(pct / 2)  # ASCII bar chart
        logger.info(f"  [{low:.1f}, {high:.1f}): {pct:5.1f}% {bar}")
        distribution[f"error_{low:.1f}_{high:.1f}"] = {
            'count': int(count),
            'percentage': float(pct)
        }
    
    return {
        'mean_error': float(errors.mean()),
        'std_error': float(errors.std()),
        'min_error': float(errors.min()),
        'max_error': float(errors.max()),
        'distribution': distribution
    }


# ============================================================================
# BASELINE COMPARISON
# ============================================================================

def compute_baseline_metrics(y_true: np.ndarray) -> Dict:
    """
    Compute metrics for simple baseline models to contextualize our model.
    
    Baselines:
        - Always predict mean rating
        - Always predict median rating
        - Always predict 3.0 (neutral)
    
    Our model is good if it beats all baselines.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("BASELINE COMPARISONS")
    logger.info(f"{'=' * 70}")
    
    mean_pred = np.full_like(y_true, np.mean(y_true), dtype=float)
    median_pred = np.full_like(y_true, np.median(y_true), dtype=float)
    constant_pred = np.full_like(y_true, 3.0, dtype=float)
    
    mean_rmse = np.sqrt(mean_squared_error(y_true, mean_pred))
    median_rmse = np.sqrt(mean_squared_error(y_true, median_pred))
    constant_rmse = np.sqrt(mean_squared_error(y_true, constant_pred))
    
    logger.info(f"Always predict mean ({np.mean(y_true):.2f}):")
    logger.info(f"  RMSE: {mean_rmse:.4f}")
    logger.info(f"\nAlways predict median ({np.median(y_true):.2f}):")
    logger.info(f"  RMSE: {median_rmse:.4f}")
    logger.info(f"\nAlways predict 3.0 (neutral):")
    logger.info(f"  RMSE: {constant_rmse:.4f}")
    
    best_baseline = min(mean_rmse, median_rmse, constant_rmse)
    
    return {
        'always_mean': {
            'value': float(np.mean(y_true)),
            'rmse': float(mean_rmse)
        },
        'always_median': {
            'value': float(np.median(y_true)),
            'rmse': float(median_rmse)
        },
        'always_neutral': {
            'value': 3.0,
            'rmse': float(constant_rmse)
        },
        'best_baseline_rmse': float(best_baseline)
    }


# ============================================================================
# SEGMENT ANALYSIS
# ============================================================================

def analyze_by_user_engagement(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame
) -> Dict:
    """
    Analyze prediction accuracy across user engagement levels.
    
    Insight: We might predict
    - Well for users with many ratings (high engagement)
    - Poorly for users with few ratings (cold start)
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("PERFORMANCE BY USER ENGAGEMENT")
    logger.info(f"{'=' * 70}")
    
    # Add predictions to test dataframe
    test_df_copy = test_df.copy()
    test_df_copy['error'] = np.abs(y_true - y_pred)
    
    # Count ratings per user
    user_ratings = test_df['user_id'].value_counts()
    test_df_copy['user_engagement'] = test_df_copy['user_id'].map(user_ratings)
    
    # Segment by engagement level
    low_engagement = test_df_copy[test_df_copy['user_engagement'] <= 5]
    medium_engagement = test_df_copy[
        (test_df_copy['user_engagement'] > 5) &
        (test_df_copy['user_engagement'] <= 20)
    ]
    high_engagement = test_df_copy[test_df_copy['user_engagement'] > 20]
    
    segments = {
        'low': low_engagement,
        'medium': medium_engagement,
        'high': high_engagement
    }
    
    results = {}
    for segment_name, segment_df in segments.items():
        if len(segment_df) > 0:
            mae = segment_df['error'].mean()
            logger.info(f"{segment_name.upper()}: {len(segment_df)} samples, MAE={mae:.4f}")
            results[segment_name] = {
                'n_samples': int(len(segment_df)),
                'mae': float(mae),
                'mean_engagement': float(segment_df['user_engagement'].mean())
            }
    
    return results