#!/bin/bash
# Run K-NN parameter sweep with MLflow

set -e

echo "Starting MLflow experiment sweep..."

# Ensure venv activated
source .venv/bin/activate

# Check MLflow server running
echo "Verifying MLflow server..."
curl -s http://localhost:5001/health > /dev/null || {
    echo "✗ MLflow server not running!"
    echo "Start it with: mlflow server --host 127.0.0.1 --port 5001"
    exit 1
}

echo "✓ MLflow server accessible"

# Run sweep
python << 'EOF'
import pandas as pd
import numpy as np
from src.sweep_experiments import run_parameter_sweep, identify_best_run
from sklearn.model_selection import train_test_split

# Load clean data
ratings_df = pd.read_csv('data/processed/ratings_clean.csv')

# Load features (for similarity matrix)
import joblib
features = joblib.load('models/rating_features.pkl')

# Create target: average rating per user
y = ratings_df.groupby('user_id')['rating'].mean()

# Split users into train/test
from sklearn.model_selection import train_test_split
train_users, test_users = train_test_split(
    features.user_ids, 
    test_size=0.2, 
    random_state=42
)

# Create train/test sets with matching indices
X_train = features.ratings_matrix.loc[train_users].values
X_test = features.ratings_matrix.loc[test_users].values
y_train = pd.DataFrame({'rating': y.loc[train_users].values})
y_test = pd.DataFrame({'rating': y.loc[test_users].values})

# Run sweep K = [3, 5, 10, 15, 20]
k_values = [3, 5, 10, 15, 20]
results = run_parameter_sweep(
    k_values=k_values,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    experiment_name="movielens_knn_sweep"
)

# Identify best
best_k, best_result = identify_best_run(results, metric="rmse")
print(f"\n{'='*60}")
print(f"Best K: {best_k}")
print(f"RMSE: {best_result['rmse']:.3f}")
print(f"MAE: {best_result['mae']:.3f}")
print(f"Coverage: {best_result['coverage']:.1%}")
print(f"Run ID: {best_result['run_id']}")
print(f"{'='*60}")

# Save results
import json
with open('evaluations/experiment_sweep_results.json', 'w') as f:
    json.dump({str(k): result for k, result in results.items()}, f, indent=2)

print(f"✓ Results saved to evaluations/experiment_sweep_results.json")
EOF

echo "✓ Sweep complete!"

