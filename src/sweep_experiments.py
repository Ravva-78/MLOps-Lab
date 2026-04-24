"""Orchestrate K-NN parameter sweep with MLflow tracking."""

import mlflow
import time
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, List
from src.mlflow_tracking import (
    initialize_mlflow_experiment,
    log_model_parameters,
    log_model_metrics,
    log_model_artifact,
    log_run_tags
)

logger = logging.getLogger(__name__)


def train_and_evaluate_knn(
    k_value: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame
) -> Dict[str, float]:
    """
    Train K-NN model and compute metrics.
    
    Args:
        k_value: Number of neighbors
        X_train: Training features
        X_test: Test features
        y_train: Training ratings (user_id, movie_id, rating)
        y_test: Test ratings
    
    Returns:
        Dict with metrics: {'rmse': ..., 'mae': ..., 'coverage': ...}
    """
    start_time = time.time()
    
    try:
        # Train K-NN
        model = NearestNeighbors(
            n_neighbors=min(k_value, len(X_train)),
            metric='cosine',
            algorithm='brute'
        )
        model.fit(X_train)
        
        # Predict: For each test point, avg rating of K nearest neighbors
        distances, indices = model.kneighbors(X_test)
        
        # Handle case where distance = 0 (user in training set)
        predictions = []
        for neighbor_indices in indices:
            neighbor_ratings = y_train.iloc[neighbor_indices]['rating'].values
            avg_rating = np.mean(neighbor_ratings) if len(neighbor_ratings) > 0 else 3.0
            predictions.append(avg_rating)
        
        predictions = np.array(predictions)
        actual = y_test['rating'].values
        
        # Compute metrics
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        mae = np.mean(np.abs(predictions - actual))
        coverage = np.sum(predictions >= 0.5) / len(predictions)
        
        training_time = time.time() - start_time
        
        return {
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'training_time': training_time,
            'model': model
        }
    
    except Exception as e:
        logger.error(f"Error training K={k_value}: {e}")
        raise


def run_parameter_sweep(
    k_values: List[int],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    experiment_name: str = "movielens_knn_sweep"
) -> Dict[str, Dict]:
    """
    Run K-NN training for multiple K values, log each to MLflow.
    
    Args:
        k_values: List of K values to try (e.g., [3, 5, 10, 15, 20])
        X_train, X_test, y_train, y_test: Training/test data
        experiment_name: MLflow experiment name
    
    Returns:
        Dict mapping k_value -> run_dict with metrics, run_id, model
    
    Example:
        results = run_parameter_sweep(
            k_values=[3, 5, 10, 15, 20],
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
        # results[5] = {'rmse': 0.889, 'mae': 0.681, 'run_id': 'abc123', ...}
    """
    # Initialize experiment
    initialize_mlflow_experiment(experiment_name)
    
    results = {}
    
    logger.info(f"Starting K-NN sweep with K values: {k_values}")
    
    for k in k_values:
        # Start new run for this K value
        with mlflow.start_run(run_name=f"k_{k}"):
            try:
                logger.info(f"\n--- Training K={k} ---")
                
                # Train and evaluate
                metrics = train_and_evaluate_knn(
                    k, X_train, X_test, y_train, y_test
                )
                
                # Log to MLflow
                log_model_parameters(k_value=k)
                log_model_metrics(
                    rmse=metrics['rmse'],
                    mae=metrics['mae'],
                    coverage=metrics['coverage'],
                    training_time_seconds=metrics['training_time']
                )
                
                # Save model
                model_path = f"models/knn_k{k}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(metrics['model'], f)
                log_model_artifact(model_path)
                
                # Tag run
                run_id = mlflow.active_run().info.run_id
                log_run_tags(run_id, {
                    "dataset": "movielens_100k",
                    "k_value": str(k),
                    "status": "completed"
                })
                
                # Store result
                results[k] = {
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'coverage': metrics['coverage'],
                    'training_time': metrics['training_time'],
                    'run_id': run_id,
                    'model_path': model_path
                }
                
                logger.info(f"✓ K={k}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, "
                           f"Coverage={metrics['coverage']:.1%}, Time={metrics['training_time']:.2f}s")
            
            except Exception as e:
                logger.error(f"✗ K={k} failed: {e}")
                mlflow.set_tag("status", "failed")
    
    return results


def identify_best_run(results: Dict, metric: str = "rmse") -> Tuple[int, Dict]:
    """
    Identify best K value based on metric (default: RMSE).
    
    Args:
        results: Dict from run_parameter_sweep
        metric: Metric to optimize ('rmse', 'mae', 'coverage')
    
    Returns:
        Tuple of (best_k, best_metrics_dict)
    
    Example:
        best_k, best_result = identify_best_run(results, metric="rmse")
        print(f"Best K: {best_k} with RMSE: {best_result['rmse']:.3f}")
    """
    if metric == "rmse":
        best_k = min(results, key=lambda k: results[k]['rmse'])
    elif metric == "mae":
        best_k = min(results, key=lambda k: results[k]['mae'])
    elif metric == "coverage":
        best_k = max(results, key=lambda k: results[k]['coverage'])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return best_k, results[best_k]