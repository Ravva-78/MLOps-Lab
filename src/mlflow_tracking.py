"""MLflow tracking utilities for experiment management."""

import mlflow
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def initialize_mlflow_experiment(
    experiment_name: str,
    tracking_uri: str = "http://localhost:5001"
) -> int:
    """
    Initialize MLflow experiment and return experiment ID.
    
    Args:
        experiment_name: Name for this experiment (e.g., "movielens_knn_sweep")
        tracking_uri: MLflow server URL
    
    Returns:
        Experiment ID (integer)
    
    Raises:
        MlflowException: If MLflow server not accessible
    
    Example:
        exp_id = initialize_mlflow_experiment("movielens_knn_sweep")
        # Subsequent runs will be logged to this experiment
    """
    try:
        # Set tracking server
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Create new experiment
            exp_id = mlflow.create_experiment(experiment_name)
            logger.info(f"✓ Created experiment: {experiment_name} (ID: {exp_id})")
        else:
            exp_id = experiment.experiment_id
            logger.info(f"✓ Using existing experiment: {experiment_name} (ID: {exp_id})")
        
        # Set as active
        mlflow.set_experiment(experiment_name)
        
        return exp_id
    
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}")
        raise


def log_model_parameters(
    k_value: int,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Log model hyperparameters to current run.
    
    Args:
        k_value: Number of neighbors for K-NN
        test_size: Train/test split ratio
        random_state: Random seed for reproducibility
    
    Example:
        mlflow.start_run()
        log_model_parameters(k_value=5)
        # MLflow tracks these for this run
    """
    mlflow.log_param("k_neighbors", k_value)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("algorithm", "k_neighbors")
    logger.info(f"✓ Logged parameters: k={k_value}, test_size={test_size}")


def log_model_metrics(
    rmse: float,
    mae: float,
    coverage: float,
    training_time_seconds: float
) -> None:
    """
    Log model performance metrics to current run.
    
    Args:
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        coverage: Coverage (% of recommendations > 0.5 rating)
        training_time_seconds: Time spent training in seconds
    
    Example:
        log_model_metrics(rmse=0.889, mae=0.681, coverage=0.87, training_time_seconds=2.1)
    """
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("coverage", coverage)
    mlflow.log_metric("training_time_seconds", training_time_seconds)
    logger.info(f"✓ Logged metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, Coverage={coverage:.1%}")


def log_model_artifact(
    model_path: str,
    artifact_name: str = "model"
) -> None:
    """
    Log model file as artifact to MLflow.
    
    Args:
        model_path: Path to model pickle file
        artifact_name: Name in MLflow (default: "model")
    
    Example:
        log_model_artifact("models/knn_k5.pkl")
    """
    try:
        mlflow.log_artifact(model_path, artifact_path=artifact_name)
        logger.info(f"✓ Logged artifact: {model_path}")
    except Exception as e:
        logger.error(f"Failed to log artifact {model_path}: {e}")
        raise


def get_mlflow_client():
    """Get MLflow client for advanced operations."""
    return mlflow.MlflowClient()


def log_run_tags(run_id: str, tags: Dict[str, str]) -> None:
    """
    Log custom tags to a run for organization.
    
    Args:
        run_id: MLflow run ID
        tags: Dict of tag key-value pairs
    
    Example:
        log_run_tags(run_id, {"dataset": "movielens_100k", "status": "best"})
    """
    client = get_mlflow_client()
    for key, value in tags.items():
        client.set_tag(run_id, key, str(value))
    logger.info(f"✓ Tagged run {run_id}: {tags}")