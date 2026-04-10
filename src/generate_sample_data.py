"""Generate synthetic MovieLens-like ratings for development."""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_ratings(
    n_ratings: int = 2000,
    n_users: int = 189,
    n_movies: int = 100,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic MovieLens-like ratings.
    
    Args:
        n_ratings: Number of ratings to generate
        n_users: Number of unique users
        n_movies: Number of unique movies
        random_seed: For reproducibility
    
    Returns:
        DataFrame with columns: user_id, movie_id, rating, timestamp
    
    Notes:
        - Ratings are sparse (~95% missing values)
        - Timestamps are in Unix format (1995-2005 range)
        - Ratings follow normal distribution centered at 3.5
    """
    
    np.random.seed(random_seed)
    
    # Generate random user-movie pairs
    user_ids = np.random.randint(1, n_users + 1, size=n_ratings)
    movie_ids = np.random.randint(1, n_movies + 1, size=n_ratings)
    
    # Ratings: normal distribution, clipped to [1, 5]
    ratings = np.random.normal(loc=3.5, scale=1.0, size=n_ratings)
    ratings = np.clip(ratings, 1.0, 5.0)
    
    # Timestamps: Unix time from 1995 to 2005
    start_timestamp = int(pd.Timestamp('1995-01-01').timestamp())
    end_timestamp = int(pd.Timestamp('2005-01-01').timestamp())
    timestamps = np.random.randint(start_timestamp, end_timestamp, size=n_ratings)
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # Sort by timestamp for realism
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def main():
    """Generate and save synthetic data."""
    
    print("Generating synthetic MovieLens ratings...")
    
    # Generate data
    df = generate_synthetic_ratings(
        n_ratings=2000,
        n_users=189,
        n_movies=100
    )
    
    # Create directory
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # Save
    output_path = 'data/raw/ratings.csv'
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"✓ Generated {len(df)} ratings")
    print(f"✓ Unique users: {df['user_id'].nunique()}")
    print(f"✓ Unique movies: {df['movie_id'].nunique()}")
    print(f"✓ Sparsity: {(1 - len(df) / (df['user_id'].nunique() * df['movie_id'].nunique())) * 100:.1f}%")
    print(f"✓ Saved to {output_path}")
    
    # Show sample
    print("\nFirst 5 ratings:")
    print(df.head())


if __name__ == '__main__':
    main()