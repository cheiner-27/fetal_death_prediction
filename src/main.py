import sys
import logging
from feature_engineering.build_features import main as build_features_main
from models.train_model import main as train_model_main
from models.predict_model import main as predict_model_main

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting the data pipeline...")

    # Step 1: Feature Engineering
    logger.info("Step 1: Building features...")
    build_features_main()

    # Step 2: Train Model
    logger.info("Step 2: Training model...")
    train_model_main()

    # Step 3: Predict (optional or separate step)
    logger.info("Step 3: Making predictions...")
    predict_model_main()

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
