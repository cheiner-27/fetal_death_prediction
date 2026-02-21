import sys
import logging
import argparse
from feature_engineering import build_features
from models.train_model import main as train_model_main
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
                        default="both_large", 
                        help="Which set of features to use.")
    
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
        logger.info("Step 2: Training model...")
        train_model_main(args)
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
