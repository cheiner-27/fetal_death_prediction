# Fetal Death Prediction

This project aims to predict fetal death using NVSS (National Vital Statistics System) data. It provides a structured pipeline for data processing, feature engineering, and model training.

## Project Status & Recent Updates (March 2026)
- **Data Investigation**: Investigated the `DPLURAL` column across 2016-2023 datasets. Confirmed that the value `9` (representing "Unknown" or "Not Stated") is present in Fetal records but not in Natal samples.
- **Model Expansion**: Added support for **Explainable Boosting Machines (EBM)** and **AutoGluon** (AutoML) alongside XGBoost and CatBoost.
- **Refined Feature Engineering**: Improved imputation logic, including stratified medians and stochastic imputation for sensitive fields like cigarette use and maternal education.

## Project Structure

```text
fetal_death_prediction/
├── data/
│   ├── csv/                # Raw data in CSV format
│   ├── guides/             # User guides and extracts
│   └── processed/          # Processed data ready for modeling
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── main.py             # Entry point for the pipeline
│   ├── data/               # Data loading and sampling scripts
│   ├── feature_engineering/# Feature engineering and cleaning
│   │   └── build_features.py
│   └── models/             # Model training and prediction
│       ├── train_xgboost.py
│       ├── train_catboost.py
│       ├── train_autogluon.py
│       ├── train_ebm.py
│       └── predict_model.py
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cheiner-27/fetal_death_prediction.git
   cd fetal_death_prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

The entire pipeline can be executed from the `src` directory using the `main.py` script.

```bash
python src/main.py
```

### Running Experiments

You can customize the data preparation process using command-line arguments:

- **--include-reporting-flags**: Include reporting flag columns (e.g., `F_CIGS_0`, `F_RF_GDIAB`) in the dataset. By default, these are excluded.
- **--feature-set**: Choose which set of features to use. Options are:
  - `numeric`: Use numeric columns only (e.g., `CIG_0`, `MAGER`, `BMI`).
  - `recode_small`: Use "small" (less granular) recode columns (e.g., `MAGER9`, `MRACE6`) and standard recodes.
  - `recode_large`: Use "large" (more granular) recode columns (e.g., `MAGER14`, `MRACE15`) and standard recodes.
  - `both_small`: Use numeric columns + small recodes.
  - `both_large`: Use numeric columns + large recodes.
- **--model**: Specify the model to train. Options: `xgboost` (default), `catboost`, `ebm`, `autogluon`.

Example:
```bash
python src/main.py --include-reporting-flags --feature-set both_large --model autogluon
```

## Key Components

### Feature Engineering (`build_features.py`)
- Responsible for cleaning raw data from `data/csv/`.
- Handles missing values, encoding, and scaling.
- Implements stratified imputation to maintain data distribution.
- Saves the resulting dataset to `data/processed/`.

### Model Training
- **XGBoost / CatBoost / EBM**: Implements manual randomized search with Stratified K-Fold CV. Prioritizes **F2 Score** via threshold tuning to ensure high recall for fetal death cases.
- **AutoGluon**: Leverages AutoML with multi-layer stacking and bagging for high-performance ensembling.

## License
[Insert License Information]
