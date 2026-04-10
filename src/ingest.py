"""Data ingestion and validation for MovieLens ratings."""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, Dict
from src.config import RATINGS_SCHEMA, DATA_PATHS, EXPECTED_METRICS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RatingsSchemaValidator:
    """Validate ratings data against schema."""
    
    def __init__(self, schema: Dict = RATINGS_SCHEMA):
        """Initialize with schema (default from config)."""
        self.schema = schema
        self.validation_report = {}
    
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Check if all required columns exist."""
        required = set(self.schema.keys())
        present = set(df.columns)
        
        if required != present:
            missing = required - present
            extra = present - required
            logger.error(f"Column mismatch. Missing: {missing}, Extra: {extra}")
            return False
        
        logger.info(f"✓ All required columns present: {list(df.columns)}")
        return True
    
    def validate_datatypes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Convert/validate column datatypes."""
        errors = 0
        
        for col, rules in self.schema.items():
            target_dtype = rules['dtype']
            
            try:
                # Attempt type conversion
                if target_dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')
                elif target_dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                
                # Count nulls from conversion failures
                current_nulls = df[col].isnull().sum()
                if current_nulls > 0:
                    logger.warning(f"{col}: {current_nulls} values couldn't convert")
                    errors += current_nulls
                
                logger.info(f"✓ {col}: dtype={target_dtype}")
            
            except Exception as e:
                logger.error(f"Failed to convert {col} to {target_dtype}: {e}")
                return df, -1
        
        return df, errors
    
    def validate_ranges(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Check if values are within valid ranges."""
        rows_before = len(df)
        
        for col, rules in self.schema.items():
            min_val = rules['min']
            max_val = rules['max']
            
            # Filter to valid range
            mask = (df[col] >= min_val) & (df[col] <= max_val)
            invalid = (~mask).sum()
            
            if invalid > 0:
                logger.warning(f"{col}: {invalid} values out of range [{min_val}, {max_val}]")
            
            df = df[mask]
        
        removed = rows_before - len(df)
        logger.info(f"✓ Range validation: {removed} rows removed, {len(df)} remain")
        return df, removed
    
    def validate_nulls(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Check for missing values."""
        rows_before = len(df)
        
        for col, rules in self.schema.items():
            if not rules['nullable']:
                nulls = df[col].isnull().sum()
                if nulls > 0:
                    logger.warning(f"{col}: {nulls} null values (not nullable)")
                    df = df.dropna(subset=[col])
        
        removed = rows_before - len(df)
        if removed > 0:
            logger.info(f"✓ Null handling: {removed} rows removed")
        return df, removed
    
    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run all validations and return clean data + report."""
        
        logger.info(f"Starting validation on {len(df)} rows...")
        
        # Step 1: Columns
        if not self.validate_columns(df):
            return None, {'error': 'Missing required columns'}
        
        # Step 2: Datatypes
        df, dtype_errors = self.validate_datatypes(df)
        if dtype_errors < 0:
            return None, {'error': 'Datatype conversion failed'}
        
        # Step 3: Ranges
        df, range_removed = self.validate_ranges(df)
        
        # Step 4: Nulls
        df, null_removed = self.validate_nulls(df)
        
        # Create report
        self.validation_report = {
            'total_errors': dtype_errors + range_removed + null_removed,
            'dtype_errors': dtype_errors,
            'range_violations': range_removed,
            'null_violations': null_removed,
            'rows_retained': len(df)
        }
        
        logger.info(f"✓ Validation complete: {len(df)} valid rows")
        return df, self.validation_report


class RatingsLoader:
    """Load ratings from CSV and validate."""
    
    def __init__(self, filepath: str = DATA_PATHS['raw']):
        """Initialize loader with file path."""
        self.filepath = filepath
        self.validator = RatingsSchemaValidator()
        self.raw_df = None
        self.clean_df = None
    
    def load(self) -> pd.DataFrame:
        """Load CSV file."""
        if not Path(self.filepath).exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        logger.info(f"Loading from {self.filepath}...")
        
        # MovieLens original uses tab separation
        self.raw_df = pd.read_csv(
            self.filepath,
            sep='\t',  # Tab-separated (standard MovieLens format)
            dtype={'user_id': 'int', 'movie_id': 'int', 'rating': 'float', 'timestamp': 'int'}
        )
        
        logger.info(f"✓ Loaded {len(self.raw_df)} ratings")
        return self.raw_df
    
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate user-movie pairs (keep last)."""
        rows_before = len(df)
        
        # Sort by timestamp first (to keep most recent rating)
        df = df.sort_values('timestamp')
        
        # Drop duplicates based on user_id + movie_id combo (keep last)
        df = df.drop_duplicates(
            subset=['user_id', 'movie_id'],
            keep='last'
        )
        
        removed = rows_before - len(df)
        if removed > 0:
            logger.info(f"✓ Deduplication: {removed} duplicates removed")
        
        return df
    
    def validate_and_clean(self) -> Tuple[pd.DataFrame, Dict]:
        """Validate and clean the loaded data."""
        if self.raw_df is None:
            raise ValueError("Must call load() first")
        
        # Step 1: Deduplicate
        df = self.deduplicate(self.raw_df)
        
        # Step 2: Validate against schema
        df, report = self.validator.validate(df)
        
        self.clean_df = df
        return df, report
    
    def save(self, output_path: str = DATA_PATHS['processed']):
        """Save cleaned data to CSV."""
        if self.clean_df is None:
            raise ValueError("No clean data to save. Run validate_and_clean() first")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.clean_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved {len(self.clean_df)} ratings to {output_path}")
    
    def save_report(self, report_path: str = DATA_PATHS['validation_report']):
        """Save validation report to JSON."""
        if not self.validator.validation_report:
            logger.warning("No validation report to save")
            return
        
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.validator.validation_report, f, indent=2)
        
        logger.info(f"✓ Saved validation report to {report_path}")


def main():
    """Full ETL pipeline."""
    
    logger.info("=" * 60)
    logger.info("MovieLens Data Ingestion Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load
        loader = RatingsLoader()
        loader.load()
        
        # Step 2: Validate and clean
        clean_df, report = loader.validate_and_clean()
        
        # Step 3: Save outputs
        loader.save()
        loader.save_report()
        
        logger.info("=" * 60)
        logger.info("✓ Pipeline complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()