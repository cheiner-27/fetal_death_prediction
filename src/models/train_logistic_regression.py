import logging
import os
import pandas as pd
import numpy as np
import joblib
import csv
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, fbeta_score, make_scorer, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_best_threshold(y_true, y_probs, beta=2):
    """
    Finds the optimal threshold to maximize F-beta score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        fbeta_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls)
    
    # Replace NaNs with 0
    fbeta_scores = np.nan_to_num(fbeta_scores)
    
    best_idx = np.argmax(fbeta_scores)
    
    # thresholds array is 1 shorter than precisions/recalls
    if best_idx < len(thresholds):
        return thresholds[best_idx], fbeta_scores[best_idx]
    else:
        return 0.5, fbeta_scores[best_idx]

def train_logistic_regression_model(args, data_path="data/processed/final_dataset.csv", models_dir="models", reports_dir="reports"):
    """
    Trains a Logistic Regression model with pipeline for preprocessing.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"logistic_regression_{timestamp}.joblib"
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

    if "OUTCOME" not in df.columns:
        logger.error("Target column 'OUTCOME' not found in dataset.")
        return

    X = df.drop(columns=["OUTCOME"])
    y = df["OUTCOME"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    logger.info(f"Categorical columns: {len(categorical_cols)}")
    logger.info(f"Numerical columns: {len(numerical_cols)}")

    # Split data: Train (80%), Test (20%)
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # Define Preprocessing Pipeline
    logger.info("Defining preprocessing pipeline...")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for easier debugging/inspection if needed
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Define Model Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')) 
    ])

    # Hyperparameter Tuning
    logger.info("Starting RandomizedSearchCV for Hyperparameter Tuning...")
    
    param_dist = {
        'classifier__C': loguniform(1e-4, 1e2),
        'classifier__penalty': ['l2']
    }
    
    # Custom Scorer for F2
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), 
        scoring=f2_scorer, 
        verbose=1, 
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    logger.info(f"Best Parameters: {search.best_params_}")
    logger.info(f"Best CV F2 Score: {search.best_score_:.4f}")

    # Threshold Tuning on Training Data (using cross-val predictions or just predicting on train for simplicity in this script, 
    # ideally we'd use cross_val_predict but let's stick to simple post-hoc tuning on a hold-out or the training set if valid enough)
    # Better approach: Predict probabilities on Train and tune threshold there.
    logger.info("Tuning decision threshold on training set...")
    y_train_probs = best_model.predict_proba(X_train)[:, 1]
    best_threshold, best_train_f2 = find_best_threshold(y_train, y_train_probs, beta=2)
    logger.info(f"Optimal Threshold (Train): {best_threshold:.4f} with F2 Score: {best_train_f2:.4f}")

    # Evaluation on Test Set
    logger.info("Evaluating final model on test set...")
    y_test_probs = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_probs >= best_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average=None, labels=[1])
    precision_pos = precision[0]
    recall_pos = recall[0]
    f1_pos = f1[0]
    f2_pos = fbeta_score(y_test, y_test_pred, beta=2, labels=[1], pos_label=1)
    
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')
    f2_w = fbeta_score(y_test, y_test_pred, beta=2, average='weighted')
    
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Positive Class - Precision: {precision_pos:.4f}, Recall: {recall_pos:.4f}, F1: {f1_pos:.4f}, F2: {f2_pos:.4f}")
    logger.info(f"Test Weighted F2 Score (at {best_threshold:.4f} threshold): {f2_w:.4f}")
    logger.info("Classification Report:\n" + report)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Save Model
    logger.info(f"Saving model to {model_path}...")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the pipeline and the threshold
    model_data = {
        'model': best_model, # This is the full pipeline
        'threshold': best_threshold,
        'feature_names': list(X.columns) # Original feature names
    }
    joblib.dump(model_data, model_path)
    logger.info("Model saved successfully.")

    # --- Generate Summary Report ---
    logger.info("Generating summary report...")
    
    # Extract coefficients for feature importance
    try:
        classifier = best_model.named_steps['classifier']
        preprocessor = best_model.named_steps['preprocessor']
        
        # Get feature names after one-hot encoding
        ohe_categories = preprocessor.named_transformers_['cat']['onehot'].categories_
        new_cat_features = []
        for i, col in enumerate(categorical_cols):
            for cat in ohe_categories[i]:
                new_cat_features.append(f"{col}_{cat}")
        
        all_features = numerical_cols + new_cat_features
        
        coefs = classifier.coef_[0]
        feat_imp_df = pd.DataFrame({"Feature": all_features, "Coefficient": coefs})
        feat_imp_df["Abs_Coefficient"] = feat_imp_df["Coefficient"].abs()
        top_20_features = feat_imp_df.sort_values(by="Abs_Coefficient", ascending=False).head(20)["Feature"].tolist()
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        top_20_features = "unavailable"

    report_data = {
        "Timestamp": timestamp,
        "Model_Path": model_path,
        "Feature_Set": getattr(args, "feature_set", "unknown"),
        "Include_Reporting_Flags": getattr(args, "include_reporting_flags", "unknown"),
        "Scoring_Metric": "f2_tuned_logistic_regression",
        "Eval_Metric": "f2",
        "Accuracy": round(accuracy, 4),
        "Best_Threshold": round(best_threshold, 4),
        "F2_Pos": round(f2_pos, 4),
        "Precision_Pos": round(precision_pos, 4),
        "Recall_Pos": round(recall_pos, 4),
        "F1_Pos": round(f1_pos, 4),
        "F2_Weighted": round(f2_w, 4),
        "Precision_Weighted": round(precision_w, 4),
        "Recall_Weighted": round(recall_w, 4),
        "F1_Weighted": round(f1_w, 4),
        "Best_Params": str(search.best_params_),
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
        
    logger.info(f"Report updated with results in {report_file}")

def main(args=None):
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="Train Logistic Regression Model")
        parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering check")
        parser.add_argument("--feature-set", type=str, default="standalone_default", help="Name of feature set")
        parser.add_argument("--include-reporting-flags", action="store_true", help="Include reporting flags")
        
        args = parser.parse_args()
        args.model = "logistic_regression"
    
    train_logistic_regression_model(args)

if __name__ == "__main__":
    main()
