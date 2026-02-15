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
- Loads processed data.
- Trains the prediction model (e.g., Random Forest, XGBoost).
- Evaluates performance and saves the model artifact to the `models/` directory.

### Prediction (`predict_model.py`)
- Loads a trained model.
- Generates predictions on new data.

## Placeholders & Future Work
- [ ] Implement specific cleaning logic in `build_features.py`.
- [ ] Define the model architecture in `train_model.py`.
- [ ] Add unit tests for data transformations.
- [ ] Integrate hyperparameter tuning.

## License
[Insert License Information]
