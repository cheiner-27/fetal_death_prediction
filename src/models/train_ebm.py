import logging
import os
import pandas as pd
import numpy as np
import joblib
import csv
import itertools
import random
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, fbeta_score, precision_recall_curve
from interpret.glassbox import ExplainableBoostingClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_best_threshold(y_true, y_probs, beta=2):
    """
    Finds the optimal threshold to maximize F-beta score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Avoid division by zero
    fbeta_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-9)
    best_idx = np.argmax(fbeta_scores)
    
    # thresholds array is 1 shorter than precisions/recalls
    if best_idx < len(thresholds):
        return thresholds[best_idx], fbeta_scores[best_idx]
    else:
        return 0.5, fbeta_scores[best_idx]

def train_ebm_model(args, data_path="data/processed/final_dataset.csv", models_dir="models", reports_dir="reports"):
    """
    Trains an Explainable Boosting Machine (EBM) with manual CV and threshold tuning per fold,
    reports positive class metrics, and uses randomized search for hyperparameters.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"ebm_{timestamp}.joblib"
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

    # EBM handles categorical features natively, but for consistency with the existing pipeline (OHE), 
    # we will stick to the same preprocessing.
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        logger.info(f"Encoding categorical columns: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    feature_names = X.columns.tolist()

    # Split data: Train (80%), Test (20%)
    logger.info("Splitting data into train and test sets...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Calculate class weight based on full training set
    # Note: EBM doesn't have a direct 'scale_pos_weight' parameter in the same way XGBoost/CatBoost do,
    # but we can pass sample weights to fit() if needed. However, EBM is generally robust.
    # We will skip manual class weighting in params and rely on EBM's internal handling or add if strictly needed.
    # There is no direct "scale_pos_weight" init param for EBM.
    
    neg_count = np.sum(y_train_full == 0)
    pos_count = np.sum(y_train_full == 1)
    
    logger.info(f"Class Balance (Train Full) - Negative (0): {neg_count}, Positive (1): {pos_count}")

    # Manual Randomized Search with CV and Per-Fold Threshold Tuning
    # EBM Hyperparameters
    param_grid = {
        'max_bins': [64],
        'interactions': [0],
        'outer_bags': [8],
        'inner_bags': [0],
        'learning_rate': [0.01, 0.05],
        'min_samples_leaf': [2, 5]
    }
    
    n_splits = 5
    n_iter = 10 # Reduced as EBM can be computationally intensive with interactions
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if len(param_combinations) > n_iter:
        logger.info(f"Sampling {n_iter} combinations from total {len(param_combinations)}...")
        random.seed(42)
        param_combinations = random.sample(param_combinations, n_iter)
    
    best_avg_f2 = -1
    best_params = None
    best_oof_probs = None

    logger.info(f"Starting Manual Randomized Search with {len(param_combinations)} combinations and {n_splits}-fold CV...")

    for i, params in enumerate(param_combinations):
        logger.info(f"Iteration {i+1}/{len(param_combinations)} - Testing params: {params}")
        fold_f2_scores = []
        oof_probs = np.zeros(len(X_train_full))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_full, y_train_full)):
            X_fold_train, X_fold_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_fold_train, y_fold_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
            
            model = ExplainableBoostingClassifier(
                **params,
                random_state=42,
                n_jobs=-1  # Use all cores
            )
            
            # EBM handles its own validation split for early stopping internally (default 15%)
            # We just fit on the fold's training data.
            model.fit(X_fold_train, y_fold_train)
            
            # Predict probabilities for OOF
            probs = model.predict_proba(X_fold_val)[:, 1]
            oof_probs[val_idx] = probs
            
            # Find best threshold for THIS fold
            _, fold_best_f2 = find_best_threshold(y_fold_val, probs, beta=2)
            
            fold_f2_scores.append(fold_best_f2)
            
            logger.info(f"Fold {fold} - Fold Best F2: {fold_best_f2:.4f}")

        avg_f2 = np.mean(fold_f2_scores)
        logger.info(f"Avg CV F2 Score: {avg_f2:.4f}")
        
        if avg_f2 > best_avg_f2:
            best_avg_f2 = avg_f2
            best_params = params
            best_oof_probs = oof_probs

    logger.info(f"Overall Best CV F2 Score: {best_avg_f2:.4f}")
    logger.info(f"Best Parameters: {best_params}")

    # Final Threshold Tuning
    best_threshold, best_oof_f2 = find_best_threshold(y_train_full, best_oof_probs, beta=2)
    logger.info(f"Optimal Threshold (from OOF): {best_threshold:.4f} with F2 Score: {best_oof_f2:.4f}")

    # Final Model Training
    logger.info(f"Training final model on all training data...")
    final_model = ExplainableBoostingClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train_full, y_train_full)

    # Evaluation
    logger.info("Evaluating final model on test set...")
    y_test_probs = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_probs > best_threshold).astype(int)
    
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

    # Save Model
    logger.info(f"Saving model to {model_path}...")
    os.makedirs(models_dir, exist_ok=True)
    model_data = {
        'model': final_model,
        'threshold': best_threshold,
        'feature_names': feature_names
    }
    joblib.dump(model_data, model_path)

    # --- Generate Summary Report ---
    # EBM global explanation (feature importance)
    try:
        ebm_global = final_model.explain_global()
        feature_importances = ebm_global.data()['scores']
        feature_names_ebm = ebm_global.data()['names']
        
        # Sort and get top 20
        # zip, sort by score descending
        sorted_features = sorted(zip(feature_names_ebm, feature_importances), key=lambda x: x[1], reverse=True)
        top_20_features = [f[0] for f in sorted_features[:20]]
    except Exception as e:
        logger.warning(f"Could not extract EBM feature importances: {e}")
        top_20_features = "N/A"

    report_data = {
        "Timestamp": timestamp,
        "Model_Path": model_path,
        "Feature_Set": getattr(args, "feature_set", "unknown"),
        "Include_Reporting_Flags": getattr(args, "include_reporting_flags", "unknown"),
        "Scoring_Metric": "f2_per_fold_threshold_random_search_ebm",
        "Eval_Metric": "auc", # EBM optimizes AUC/LogLoss by default
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
        "Best_Params": str(best_params),
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
        class Args:
            feature_set = "standalone_default"
            include_reporting_flags = False
            model = "ebm"
        args = Args()
    
    train_ebm_model(args)

if __name__ == "__main__":
    main()
