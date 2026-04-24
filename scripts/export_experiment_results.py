"""Export MLflow experiment results to CSV for analysis."""

import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5001")

# Get experiment
experiment = mlflow.get_experiment_by_name("movielens_knn_sweep")
experiment_id = experiment.experiment_id

# Search runs
runs = mlflow.search_runs(experiment_ids=[experiment_id])

# Extract metrics and params
comparison_data = []
for _, run in runs.iterrows():
    comparison_data.append({
        'run_id': run['run_id'],
        'k_value': run['params.k_neighbors'],
        'rmse': run['metrics.rmse'],
        'mae': run['metrics.mae'],
        'coverage': run['metrics.coverage'],
        'training_time': run['metrics.training_time_seconds'],
        'status': run['status']
    })

# Convert to DataFrame
df = pd.DataFrame(comparison_data)

# Sort by RMSE
df_sorted = df.sort_values('rmse')

# Save to CSV
df_sorted.to_csv('evaluations/experiment_comparison.csv', index=False)

# Print summary
print("\nExperiment Results Summary:")
print("="*80)
print(df_sorted.to_string(index=False))
print("="*80)
print(f"\nBest K: {df_sorted.iloc[0]['k_value']} with RMSE: {df_sorted.iloc[0]['rmse']:.3f}")