import argparse
import json
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.train import KNNRecommendationModel
from src.tune_hyperparameters import tune_k_parameter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """
    Complete model training pipeline.
    
    Steps:
        1. Load feature store (Lab 5 output)
        2. Load clean ratings (Lab 4 output)
        3. Train/validation split
        4. (Optional) Hyperparameter tuning on validation set
        5. Train final model on all training data
        6. Evaluate on validation set
        7. Save model and metadata
    """
    logger.info("=" * 70)
    logger.info("MODEL TRAINING PIPELINE: K-NN Recommendation Model")
    logger.info("=" * 70)
    
    try:
        # Step 1: Load feature store
        logger.info(f"\nStep 1: Loading feature store from {args.features_path}")
        features = joblib.load(args.features_path)
        logger.info(f"  ✓ Features loaded")
        logger.info(f"  Users: {len(features.user_ids)}")
        logger.info(f"  Movies: {len(features.movie_ids)}")
        
        # Step 2: Load ratings
        logger.info(f"\nStep 2: Loading ratings from {args.ratings_path}")
        ratings_df = pd.read_csv(args.ratings_path)
        logger.info(f"  ✓ Loaded {len(ratings_df)} ratings")
        
        # Step 3: Train/validation split
        logger.info(f"\nStep 3: Splitting into train/validation (80/20)")
        n = len(ratings_df)
        np.random.seed(42)  # Reproducible splits
        train_idx = np.random.choice(n, int(0.8 * n), replace=False)
        val_idx = np.setdiff1d(np.arange(n), train_idx)
        
        train_df = ratings_df.iloc[train_idx].reset_index(drop=True)
        val_df = ratings_df.iloc[val_idx].reset_index(drop=True)
        
        logger.info(f"  Train set: {len(train_df)} samples")
        logger.info(f"  Val set:   {len(val_df)} samples")
        
        # Step 4: Hyperparameter tuning (optional)
        if args.tune:
            logger.info(f"\nStep 4: Tuning K parameter")
            best_k, tuning_results = tune_k_parameter(
                features,
                train_df,
                val_df,
                k_values=args.k_values
            )
            k_to_use = best_k
            
            # Save tuning results
            tuning_log = {
                'tuning_results': tuning_results,
                'best_k': int(best_k)
            }
            tuning_file = f'{args.model_dir}/tuning_results.json'
            with open(tuning_file, 'w') as f:
                json.dump(tuning_log, f, indent=2)
            logger.info(f"  ✓ Saved tuning results to {tuning_file}")
        else:
            logger.info(f"\nStep 4: Using K={args.k}")
            k_to_use = args.k
        
        # Step 5: Train final model
        logger.info(f"\nStep 5: Training final model (K={k_to_use})")
        model = KNNRecommendationModel(
            k=int(k_to_use),
            use_similarity_weights=args.use_weights
        )
        model.fit(features, train_df)
        logger.info(f"  ✓ Model trained")
        
        # Step 6: Evaluate on validation
        logger.info(f"\nStep 6: Evaluating on validation set")
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        y_true = val_df['rating'].values
        y_pred = model.predict_batch(val_df)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        logger.info(f"  ✓ Validation metrics:")
        logger.info(f"    RMSE: {rmse:.4f}")
        logger.info(f"    MAE:  {mae:.4f}")
        
        # Step 7: Save artifacts
        logger.info(f"\nStep 7: Saving model and metadata")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = f'{args.model_dir}/model.pkl'
        model.save(model_file)
        
        # Save metadata
        metadata = {
            'model_type': 'KNNRecommendation',
            'hyperparameters': {
                'k': int(k_to_use),
                'use_similarity_weights': args.use_weights,
                'default_rating': model.default_rating
            },
            'training': {
                'n_train_samples': int(len(train_df)),
                'n_val_samples': int(len(val_df)),
                'split_ratio': 0.8
            },
            'evaluation': {
                'rmse': float(rmse),
                'mae': float(mae),
                'dataset': 'MovieLens (subset)'
            },
            'feature_store': args.features_path,
            'ratings_source': args.ratings_path
        }
        
        metadata_file = f'{args.model_dir}/metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✓ Saved metadata to {metadata_file}")
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"✓ TRAINING COMPLETE")
        logger.info(f"{'=' * 70}")
        
        return model, metadata
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train K-NN recommendation model'
    )
    parser.add_argument(
        '--features_path',
        default='models/rating_features.pkl',
        help='Path to fitted RatingFeatures (from Lab 5)'
    )
    parser.add_argument(
        '--ratings_path',
        default='data/ratings_clean.csv',
        help='Path to clean ratings CSV (from Lab 4)'
    )
    parser.add_argument(
        '--model_dir',
        default='models',
        help='Directory to save model and metadata'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='K value for K-NN (default: 5)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning'
    )
    parser.add_argument(
        '--k_values',
        type=int,
        nargs='+',
        default=[3, 5, 10, 15, 20],
        help='K values to try during tuning'
    )
    parser.add_argument(
        '--use_weights',
        action='store_true',
        help='Use similarity-weighted average instead of simple mean'
    )
    
    args = parser.parse_args()
    main(args)