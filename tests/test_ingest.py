"""Tests for data ingestion pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.ingest import RatingsLoader, RatingsSchemaValidator
from src.config import RATINGS_SCHEMA, DATA_PATHS


class TestRatingsSchemaValidator:
    """Test validation logic."""
    
    @pytest.fixture
    def validator(self):
        """Setup validator."""
        return RatingsSchemaValidator()
    
    @pytest.fixture
    def sample_valid_df(self):
        """Sample valid DataFrame."""
        return pd.DataFrame({
            'user_id': [1, 2, 3],
            'movie_id': [10, 20, 30],
            'rating': [4.5, 3.0, 5.0],
            'timestamp': [978300000, 978300001, 978300002]
        })
    
    def test_validate_columns(self, validator, sample_valid_df):
        """Test column validation."""
        assert validator.validate_columns(sample_valid_df) == True
    
    def test_validate_missing_column(self, validator):
        """Test with missing column."""
        df = pd.DataFrame({'user_id': [1], 'movie_id': [1], 'rating': [4.0]})
        assert validator.validate_columns(df) == False
    
    def test_validate_datatypes(self, validator, sample_valid_df):
        """Test datatype conversion."""
        df, errors = validator.validate_datatypes(sample_valid_df)
        assert errors == 0
        assert df['user_id'].dtype == 'int64'
    
    def test_validate_ranges(self, validator, sample_valid_df):
        """Test range validation."""
        df, removed = validator.validate_ranges(sample_valid_df)
        assert removed == 0
        assert len(df) == 3


class TestRatingsLoader:
    """Test loader functionality."""
    
    def test_loader_init(self):
        """Test loader initialization."""
        loader = RatingsLoader(DATA_PATHS['raw'])
        assert loader.filepath == DATA_PATHS['raw']
    
    def test_load_file_not_found(self):
        """Test loading non-existent file."""
        loader = RatingsLoader('nonexistent.csv')
        with pytest.raises(FileNotFoundError):
            loader.load()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])