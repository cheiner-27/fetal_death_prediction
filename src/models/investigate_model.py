import pandas as pd
import joblib
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_model(data_path="data/processed/final_dataset.csv", model_path="models/xgboost_model.joblib"):
    """
    Investigates a trained XGBoost model for signs of data leakage and class imbalance.
    """
    logger.info(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    
    # 1. Check Class Imbalance
    if "OUTCOME" not in df.columns:
        logger.error("Target column 'OUTCOME' not found.")
        return

    outcome_counts = df["OUTCOME"].value_counts(normalize=True)
    logger.info(f"Class Distribution (OUTCOME):\n{outcome_counts}")
    
    if outcome_counts.max() > 0.95:
        logger.warning("EXTREME CLASS IMBALANCE DETECTED! The model might just be predicting the majority class.")

    # 2. Check Feature Importance
    logger.info(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}.")
        return

    model = joblib.load(model_path)
    
    # Get feature names (handling one-hot encoding if necessary)
    # Ideally, we should have saved the feature names with the model, but we can reconstruct them
    # by applying the same preprocessing as train_model.py
    
    X = df.drop(columns=["OUTCOME"])
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    feature_names = X.columns.tolist()
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feature_imp_df = feature_imp_df.sort_values(by="Importance", ascending=False).head(20)
        
        logger.info("Top 20 Feature Importances:")
        print(feature_imp_df)
        
        # Check for potential leaks
        top_feature = feature_imp_df.iloc[0]
        if top_feature["Importance"] > 0.5:
             logger.warning(f"POTENTIAL LEAK DETECTED: Feature '{top_feature['Feature']}' has dominantly high importance ({top_feature['Importance']:.4f}).")

        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=feature_imp_df)
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig("reports/figures/feature_importance_check.png")
        logger.info("Feature importance plot saved to reports/figures/feature_importance_check.png")

    else:
        logger.info("Model does not have feature_importances_ attribute (might be a pipeline or different model type).")

if __name__ == "__main__":
    investigate_model()
