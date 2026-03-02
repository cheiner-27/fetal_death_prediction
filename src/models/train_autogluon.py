import logging
import os
import pandas as pd
import numpy as np
import joblib
import csv
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, fbeta_score
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_best_threshold(y_true, y_probs, beta=2):
    best_threshold = 0.5
    best_score = -1
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta)
        if score > best_score:
            best_score = score
            best_threshold = thresh
            
    return best_threshold, best_score

def train_autogluon_model(args, data_path="data/processed/final_dataset.csv", models_dir="models", reports_dir="reports"):
    """
    Trains an AutoGluon model, tunes the threshold using calibrate_decision_threshold,
    reports positive class metrics, and appends a summary report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # AutoGluon saves its own directory, but we can store the predictor path
    ag_path = os.path.join(models_dir, f"autogluon_{timestamp}")
    
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

    # Split data: Train (85%), Test (15%)
    logger.info("Splitting data into train and test sets...")
    train_data, test_data = train_test_split(df, test_size=0.15, random_state=42, stratify=df["OUTCOME"])

    # Define custom F-beta metric for AutoGluon to optimize

    def f_beta_metric_func(y_true, y_pred):
        return fbeta_score(y_true, y_pred, beta=2)

    # AutoGluon scorer
    ag_f2_scorer = make_scorer(name='f2', score_func=f_beta_metric_func, greater_is_better=True, needs_proba=False)

    # Configuration Parameters
    time_limit = 28000  # Increasing for overnight
    presets = 'best_quality' # 'best_quality' enables bagging/stacking
    eval_metric = 'average_precision' # Optimize for PR curve area first

    logger.info(f"Starting AutoGluon training (time_limit={time_limit}s, presets={presets})...")
    
    hyperparameters = {
        'GBM': [
            {},                              # LightGBM default
            {'extra_trees': True},           # LightGBMXT
        ],          
        'CAT': {},          # CatBoost (Great for categorical features)
        'XGB': {},          # XGBoost
        'NN_TORCH': {},     # PyTorch Neural Network
        'FASTAI': {},       # FastAI Neural Network
        'RF': [
            {'criterion': 'gini'},
            {'criterion': 'entropy'},
        ],                  # Random Forest
        'XT': {},           # Extra Trees
    }

    predictor = TabularPredictor(
        label="OUTCOME", 
        eval_metric=eval_metric, 
        path=ag_path
    ).fit(
        train_data,
        time_limit=time_limit,
        num_bag_folds=8,
        hyperparameters=hyperparameters,
        presets=presets,
        num_gpus=1,
        ag_args_fit={'num_gpus': 1} # Enable GPU support
    )

    # Calibrate Decision Threshold using Manual OOF method
    logger.info("Calibrating decision threshold for F2 score using OOF predictions...")
    try:
        # Get OOF predictions (probability of positive class)
        oof_probs = predictor.get_oof_pred_proba(as_multiclass=False)
        # Align labels with OOF predictions
        oof_labels = train_data["OUTCOME"].loc[oof_probs.index]
        
        best_threshold, best_oof_f2 = find_best_threshold(oof_labels, oof_probs, beta=2)
        logger.info(f"Optimal Threshold (from OOF/Internal): {best_threshold:.4f} with F2 Score: {best_oof_f2:.4f}")
        
        # Set the threshold in the predictor for subsequent predictions
        predictor.set_decision_threshold(best_threshold)
        
    except Exception as e:
        logger.warning(f"Could not calculate OOF threshold, falling back to 0.5: {e}")
        best_threshold = 0.5
        predictor.set_decision_threshold(0.5)

    # Evaluation on Test Set
    logger.info("Evaluating final model on test set...")
    # predictor.predict uses the calibrated threshold automatically now
    y_test = test_data["OUTCOME"]
    y_test_pred = predictor.predict(test_data)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average=None, labels=[1])
    precision_pos = precision[0]
    recall_pos = recall[0]
    f1_pos = f1[0]
    f2_pos = fbeta_score(y_test, y_test_pred, beta=2, labels=[1], pos_label=1)
    
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')
    f2_w = fbeta_score(y_test, y_test_pred, beta=2, average='weighted')
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Positive Class - Precision: {precision_pos:.4f}, Recall: {recall_pos:.4f}, F1: {f1_pos:.4f}, F2: {f2_pos:.4f}")
    logger.info(f"Test Weighted F2 Score (at {best_threshold:.4f} threshold): {f2_w:.4f}")

    # Calculate Feature Importance
    top_20_features_list = "unavailable"
    try:
        logger.info("Computing feature importance (this may take a while)...")
        importance = predictor.feature_importance(
            test_data.sample(n=50000, random_state=42),  # 
            num_shuffle_sets=3
        )

        top_20_features_list = importance.head(20).index.tolist()
        logger.info(f"Top 20 features by permutation importance:\n{importance.head(20)}")
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")

    # Save threshold separately (redundant if predictor saves it, but good for reporting)
    threshold_path = os.path.join(ag_path, "custom_threshold.joblib")
    joblib.dump(best_threshold, threshold_path)

    # --- Generate Summary Report ---
    report_data = {
        "Timestamp": timestamp,
        "Model_Path": ag_path,
        "Feature_Set": getattr(args, "feature_set", "unknown"),
        "Include_Reporting_Flags": getattr(args, "include_reporting_flags", "unknown"),
        "Scoring_Metric": f"autogluon_{presets}",
        "Eval_Metric": eval_metric,
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
        "Best_Params": "See AutoGluon Leaderboard",
        "Confusion_Matrix": str(cm.tolist()),
        "Top_20_Features": str(top_20_features_list)
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
    
    # Print Leaderboard
    logger.info("AutoGluon Leaderboard:")
    print(predictor.leaderboard(test_data, silent=True))

def main(args=None):
    if args is None:
        class Args:
            feature_set = "standalone_default"
            include_reporting_flags = False
            model = "autogluon"
        args = Args()
    
    train_autogluon_model(args)

if __name__ == "__main__":
    main()
