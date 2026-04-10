import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_ratings(df):
    """Validate MovieLens schema and ranges."""
    required = {'user_id', 'movie_id', 'rating', 'timestamp'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    
    # Remove invalid ratings
    initial = len(df)
    df = df[(df['rating'] >= 1.0) & (df['rating'] <= 5.0)]
    invalid = initial - len(df)
    
    if invalid > 0:
        logger.warning(f"Removed {invalid} invalid ratings")
    
    return df

def process_ratings():
    """Load, validate, and save clean ratings."""
    logger.info("Loading raw ratings...")
    df = pd.read_csv('data/raw/ratings.csv')
    logger.info(f"Loaded {len(df)} ratings")
    
    # Validate
    df = validate_ratings(df)
    
    # Remove duplicates on (user_id, movie_id)
    before = len(df)
    df = df.drop_duplicates(subset=['user_id', 'movie_id'])
    removed = before - len(df)
    
    if removed > 0:
        logger.info(f"Removed {removed} duplicate ratings")
    
    # Save
    Path('data/processed').mkdir(exist_ok=True)
    df.to_csv('data/processed/ratings_clean.csv', index=False)
    
    logger.info(f"✓ Saved {len(df)} clean ratings")
    logger.info(f"  Users: {df['user_id'].nunique()}")
    logger.info(f"  Movies: {df['movie_id'].nunique()}")

if __name__ == '__main__':
    process_ratings()