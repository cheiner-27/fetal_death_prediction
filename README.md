# Fetal Death Prediction

This project aims to predict fetal death using NVSS (National Vital Statistics System) data. It provides a structured pipeline for data processing, feature engineering, and model training.

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
│       ├── train_model.py
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

Example:
```bash
python src/main.py --include-reporting-flags --feature-set both_large
```

This will sequentially run:
1. **Feature Engineering**: `src/feature_engineering/build_features.py`
2. **Model Training**: `src/models/train_model.py`
3. **Prediction**: `src/models/predict_model.py`

## Key Components

### Feature Engineering (`build_features.py`)
- Responsible for cleaning raw data from `data/csv/`.
- Handles missing values, encoding, and scaling.
- Saves the resulting dataset to `data/processed/`.

### Model Training (`train_model.py`)
- Loads processed data from `data/processed/final_dataset.csv`.
- Trains an **XGBoost Classifier** using `GridSearchCV` for hyperparameter optimization.
- One-hot encodes categorical variables automatically.
- Evaluates performance (Accuracy, Classification Report) on a test set (20% split).
- Saves the best model artifact to `models/xgboost_model.joblib`.

### Prediction (`predict_model.py`)
- Loads a trained model.
- Generates predictions on new data.

## Placeholders & Future Work
- [x] Implement specific cleaning logic in `build_features.py`.
- [x] Define the model architecture in `train_model.py`.
- [ ] Add unit tests for data transformations.
- [x] Integrate hyperparameter tuning.

## License
[Insert License Information]
