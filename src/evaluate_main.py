import argparse
import json
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.evaluate import (
    load_model,
    load_metadata,
    evaluate_rating_prediction,
    compute_coverage,
    analyze_sparsity,
    analyze_error_distribution,
    compute_baseline_metrics,
    analyze_by_user_engagement
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """
    Complete model evaluation pipeline.
    
    Steps:
        1. Load trained model and feature store
        2. Load test ratings
        3. Compute predictions
        4. Evaluate rating prediction accuracy
        5. Analyze catalog coverage
        6. Analyze data sparsity
        7. Analyze error distribution
        8. Compare against baselines
        9. Segment analysis
        10. Save comprehensive report
    """
    logger.info("=" * 70)
    logger.info("MODEL EVALUATION PIPELINE: Comprehensive Analysis")
    logger.info("=" * 70)
    
    try:
        # Step 1: Load artifacts
        logger.info("\nStep 1: Loading model and metadata")
        model = load_model(args.model_path)
        metadata = load_metadata(args.metadata_path)
        logger.info(f"  Model K: {metadata['hyperparameters']['k']}")
        
        # Step 2: Load test data
        logger.info(f"\nStep 2: Loading test ratings from {args.test_path}")
        test_df = pd.read_csv(args.test_path)
        logger.info(f"  Loaded {len(test_df)} test ratings")
        
        # Load full ratings for sparsity analysis
        logger.info(f"\nStep 3: Loading full ratings for sparsity analysis")
        ratings_df = pd.read_csv(args.ratings_path)
        logger.info(f"  Loaded {len(ratings_df)} total ratings")
        
        # Step 4: Generate predictions
        logger.info(f"\nStep 4: Generating model predictions")
        y_true = test_df['rating'].values
        y_pred = model.predict_batch(test_df)
        logger.info(f"  ✓ Generated {len(y_pred)} predictions")
        
        # Step 5: Evaluate rating prediction
        logger.info(f"\nStep 5: Evaluating rating prediction accuracy")
        rating_metrics = evaluate_rating_prediction(y_true, y_pred)
        
        # Step 6: Coverage analysis
        logger.info(f"\nStep 6: Analyzing catalog coverage")
        coverage_metrics = compute_coverage(model, test_df)
        
        # Step 7: Sparsity analysis
        logger.info(f"\nStep 7: Analyzing data sparsity")
        sparsity_metrics = analyze_sparsity(ratings_df, args.n_movies)
        
        # Step 8: Error distribution
        logger.info(f"\nStep 8: Analyzing error distribution")
        error_metrics = analyze_error_distribution(y_true, y_pred)
        
        # Step 9: Baseline comparison
        logger.info(f"\nStep 9: Computing baseline metrics")
        baseline_metrics = compute_baseline_metrics(y_true)
        
        # Step 10: Segment analysis
        logger.info(f"\nStep 10: Analyzing by user engagement")
        segment_metrics = analyze_by_user_engagement(y_true, y_pred, test_df)
        
        # Compile all metrics
        evaluation_report = {
            'metadata': metadata,
            'test_set': {
                'n_samples': int(len(test_df)),
                'date': pd.Timestamp.now().isoformat()
            },
            'rating_prediction': rating_metrics,
            'coverage': coverage_metrics,
            'sparsity': sparsity_metrics,
            'error_distribution': error_metrics,
            'baselines': baseline_metrics,
            'by_engagement': segment_metrics
        }
        
        # Save report
        logger.info(f"\n{'=' * 70}")
        logger.info("Step 11: Saving evaluation report")
        Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
        
        report_file = f'{args.eval_dir}/evaluation_report.json'
        with open(report_file, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        logger.info(f"✓ Saved report to {report_file}")
        
        # Print summary
        logger.info(f"\n{'=' * 70}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'=' * 70}")
        logger.info(f"RMSE:              {rating_metrics['rmse']:.4f}")
        logger.info(f"MAE:               {rating_metrics['mae']:.4f}")
        logger.info(f"Catalog Coverage:  {coverage_metrics['coverage_ratio']:.2%}")
        logger.info(f"Data Sparsity:     {sparsity_metrics['sparsity']:.2%}")
        logger.info(f"Best Baseline:     {baseline_metrics['best_baseline_rmse']:.4f}")
        
        # Assess performance
        if rating_metrics['rmse'] < baseline_metrics['best_baseline_rmse']:
            logger.info(f"✓ Model beats baseline!")
        else:
            logger.warning(f"⚠ Model doesn't beat baseline - needs improvement")
        
        logger.info(f"\n{'=' * 70}")
        logger.info("✓ EVALUATION COMPLETE")
        logger.info(f"{'=' * 70}")
        
        return evaluation_report
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate recommendation model')
    parser.add_argument(
        '--model_path',
        default='models/model.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--metadata_path',
        default='models/metadata.json',
        help='Path to model metadata'
    )
    parser.add_argument(
        '--test_path',
        default='data/processed/ratings_clean.csv',
        help='Path to test ratings'
    )
    parser.add_argument(
        '--ratings_path',
        default='data/processed/ratings_clean.csv',
        help='Path to all ratings (for sparsity analysis)'
    )
    parser.add_argument(
        '--n_movies',
        type=int,
        default=100,
        help='Number of movies in catalog'
    )
    parser.add_argument(
        '--eval_dir',
        default='evaluations',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    main(args)