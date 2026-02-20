import logging
import os
import pandas as pd
import numpy as np
import joblib
import csv
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_xgboost_model(args, data_path="data/processed/final_dataset.csv", models_dir="models", reports_dir="reports"):
    """
    Trains an XGBoost model, saves it with a timestamp, and appends a summary report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgboost_{timestamp}.joblib"
    model_path = os.path.join(models_dir, model_filename)
    
    logger.info(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}. Please run feature engineering first.")
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Failed to read data: {e}")
        return

    # Separate features and target
    if "OUTCOME" not in df.columns:
        logger.error("Target column 'OUTCOME' not found in dataset.")
        return

    X = df.drop(columns=["OUTCOME"])
    y = df["OUTCOME"]

    # Preprocessing: One-Hot Encoding
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        logger.info(f"Encoding categorical columns: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Save feature names for importance mapping
    feature_names = X.columns.tolist()

    # Split data
    logger.info("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Calculate class weight (still useful to log, or use in conjunction with SMOTE)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    logger.info(f"Class Balance - Negative (0): {neg_count}, Positive (1): {pos_count}")
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Initialize XGBoost Classifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    # Initialize SMOTE
    smote = SMOTE(random_state=42)

    # Create Pipeline
    # SMOTE happens only during training in each fold of CV
    pipeline = Pipeline([
        ('smote', smote),
        ('xgb', xgb)
    ])

    # Grid Search
    # Note: We can test scale_pos_weight=1 (since SMOTE balances classes) vs calculated weight
    param_grid = {
        'xgb__n_estimators': [100],
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.1],
        'xgb__scale_pos_weight': [1, scale_pos_weight] 
    }

    logger.info("Starting GridSearchCV with SMOTE pipeline and F1 scoring...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',  # Optimizing for F1 score of the positive class
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Best Model (Pipeline)
    best_pipeline = grid_search.best_estimator_
    logger.info(f"Best Parameters: {grid_search.best_params_}")

    # Evaluation
    logger.info("Evaluating model on test set...")
    y_pred = best_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:\n" + report)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Save Model
    logger.info(f"Saving model to {model_path}...")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    logger.info("Model saved successfully.")

    # --- Generate Summary Report ---
    logger.info("Generating summary report...")
    
    # Feature Importance (Extract from the XGBoost step in the pipeline)
    # best_pipeline.named_steps['xgb'] gives us the trained XGBClassifier
    best_xgb_model = best_pipeline.named_steps['xgb']
    importances = best_xgb_model.feature_importances_
    
    feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    top_20_features = feat_imp_df.sort_values(by="Importance", ascending=False).head(20)["Feature"].tolist()
    
    # Prepare Report Data
    report_data = {
        "Timestamp": timestamp,
        "Model_Path": model_path,
        "Feature_Set": getattr(args, "feature_set", "unknown"),
        "Include_Reporting_Flags": getattr(args, "include_reporting_flags", "unknown"),
        "Accuracy": round(accuracy, 4),
        "Precision_Weighted": round(precision, 4),
        "Recall_Weighted": round(recall, 4),
        "F1_Weighted": round(f1, 4),
        "Best_Params": str(grid_search.best_params_),
        "Confusion_Matrix": str(cm.tolist()),
        "Top_20_Features": str(top_20_features)
    }

    report_file = os.path.join(reports_dir, "experiment_results.csv")
    os.makedirs(reports_dir, exist_ok=True)
    
    file_exists = os.path.isfile(report_file)
    
    with open(report_file, mode='a', newline='') as csvfile:
        fieldnames = list(report_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        
        writer.writerow(report_data)
        
    logger.info(f"Report appended to {report_file}")

def main(args=None):
    """
    Main function to train the model.
    """
    if args is None:
        # Default dummy args if run standalone
        class Args:
            feature_set = "standalone_default"
            include_reporting_flags = False
        args = Args()

    logger.info("Starting model training pipeline...")
    train_xgboost_model(args)
    logger.info("Model training pipeline completed.")

if __name__ == "__main__":
    main()
