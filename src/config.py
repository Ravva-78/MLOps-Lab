# ============================================================================
# MovieLens 100K Dataset Schema
# ============================================================================
# Original dataset: 100,000 ratings from 943 users on 1,682 movies
# For our lab: Synthetic subset for learning (2,000 ratings, 189 users, 100 movies)
# ============================================================================

RATINGS_SCHEMA = {
    'user_id': {
        'dtype': 'int64',
        'min': 1,
        'max': 1000,  # Allow for expansion beyond initial 189 users
        'nullable': False,
        'description': 'Unique user identifier'
    },
    'movie_id': {
        'dtype': 'int64',
        'min': 1,
        'max': 1700,  # Allow for expansion beyond initial 100 movies
        'nullable': False,
        'description': 'Unique movie identifier'
    },
    'rating': {
        'dtype': 'float64',
        'min': 0.5,
        'max': 5.0,
        'nullable': False,
        'description': 'User rating on 1-5 scale (0.5 increments)'
    },
    'timestamp': {
        'dtype': 'int64',
        'min': 700000000,  # ~2001 (Unix epoch)
        'max': 2000000000,  # ~2033 (future-proof)
        'nullable': False,
        'description': 'Unix timestamp of rating'
    }
}
# Expected row counts (for comparing before/after)
EXPECTED_METRICS = {
    'raw_rows': 2000,
    'min_clean_rows': 1800,  # At least 90% valid
    'target_sparsity': 0.90   # ~10% density (1940/(189*100))
}

# File paths relative to project root
DATA_PATHS = {
    'raw': 'data/raw/ratings.csv',
    'processed': 'data/processed/ratings_clean.csv',
    'validation_report': 'evaluations/validation_report.json'
}