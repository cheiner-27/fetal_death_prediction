import sys
import logging
import argparse
from feature_engineering import build_features
from models.train_xgboost import main as train_xgboost_main
from models.train_catboost import main as train_catboost_main
from models.train_ebm import main as train_ebm_main
from models.train_autogluon import main as train_autogluon_main
from models.train_logistic_regression import main as train_logistic_regression_main
from models.predict_model import main as predict_model_main

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the Fetal Death Prediction pipeline.")
    
    # Add arguments for build_features
    parser.add_argument("--include-reporting-flags", action="store_true", help="Include reporting flags in the dataset.")
    parser.add_argument("--feature-set", 
                        choices=["numeric", "recode_small", "recode_large", "both_small", "both_large"], 
                        default="both_small", 
                        help="Which set of features to use.")
    
    # Model selection
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "catboost", "autogluon", "ebm", "logistic_regression"], help="Model to train (default: xgboost).")
    
    # Arguments for other steps
    parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering step.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training step.")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction step.")

    args = parser.parse_args()

    logger.info("Starting the data pipeline...")

    # Step 1: Feature Engineering
    if not args.skip_features:
        logger.info("Step 1: Building features...")
        build_features.main(args)
    else:
        logger.info("Skipping feature engineering.")

    # Step 2: Train Model
    if not args.skip_train:
        logger.info(f"Step 2: Training {args.model} model...")
        if args.model == "xgboost":
            train_xgboost_main(args)
        elif args.model == "catboost":
            train_catboost_main(args)
        elif args.model == "autogluon":
            train_autogluon_main(args)
        elif args.model == "ebm":
            train_ebm_main(args)
        elif args.model == "logistic_regression":
            train_logistic_regression_main(args)
    else:
        logger.info("Skipping training.")

    # Step 3: Predict (optional or separate step)
    if not args.skip_predict:
        logger.info("Step 3: Making predictions...")
        predict_model_main()
    else:
        logger.info("Skipping predictions.")

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
