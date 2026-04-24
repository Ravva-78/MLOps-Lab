"""Test MLflow tracking functionality."""

import pytest
import mlflow
import os
from src.sweep_experiments import run_parameter_sweep, identify_best_run
import numpy as np
import pandas as pd


@pytest.fixture
def sample_data():
    """Create small sample data for testing."""
    X_train = np.random.randn(20, 10)
    X_test = np.random.randn(5, 10)
    y_train = pd.DataFrame({
        'rating': np.random.uniform(0.5, 5.0, 20),
        'user_id': np.arange(1, 21),
        'movie_id': np.random.randint(1, 100, 20)
    })
    y_test = pd.DataFrame({
        'rating': np.random.uniform(0.5, 5.0, 5),
        'user_id': np.arange(21, 26),
        'movie_id': np.random.randint(1, 100, 5)
    })
    return X_train, X_test, y_train, y_test


def test_experiment_runs(sample_data):
    """Test that experiment run completes successfully."""
    X_train, X_test, y_train, y_test = sample_data
    
    mlflow.set_tracking_uri("http://localhost:5001")
    
    results = run_parameter_sweep(
        k_values=[3, 5],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        experiment_name="test_experiment"
    )
    
    assert len(results) == 2
    assert 3 in results
    assert 5 in results


def test_metrics_in_valid_range(sample_data):
    """Test that metrics are in expected ranges."""
    X_train, X_test, y_train, y_test = sample_data
    
    results = run_parameter_sweep(
        k_values=[5],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        experiment_name="test_metrics_range"
    )
    
    result = results[5]
    
    # RMSE should be positive
    assert result['rmse'] >= 0
    # Coverage should be between 0 and 1
    assert 0 <= result['coverage'] <= 1
    # Training time should be positive
    assert result['training_time'] > 0


def test_best_run_identification(sample_data):
    """Test identifying best run."""
    X_train, X_test, y_train, y_test = sample_data
    
    results = run_parameter_sweep(
        k_values=[3, 5, 10],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        experiment_name="test_best_run"
    )
    
    best_k, best_result = identify_best_run(results, metric="rmse")
    
    # Best K should be one of the tested values
    assert best_k in [3, 5, 10]
    # Best result should have RMSE
    assert 'rmse' in best_result
    assert 'run_id' in best_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])