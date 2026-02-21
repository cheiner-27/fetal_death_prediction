import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and Column Groups
DATA_LEAK_COLUMNS = [
    "BWTR4", "COMBGEST", "DBWT", "ME_PRES", "OEGest_Comb",  # Baby related
    "ATTEND", "BFACIL", "BFACIL3", "DMETH_REC", "ME_ROUT", "ME_TRIAL", "MM_AICU", "MM_RUPT", "RDMETH_REC", "YEAR",  # Delivery related
    "F_MM_AICU", "OBGEST_FLG",  # Reporting flags
    "CIG_3", "DLMP_YY",
    "LBO_REC",  # Perfect predictor for fetal death (LBO_REC=0)
    "SEX", "WIC",  # Highly correlated 'Unknown' status
    "IMP_SEX"  # Imputed Sex Flag (proxy for SEX leak)
]

REPORTING_FLAGS = [
    "F_CIGS_0", "F_CIGS_1", "F_CIGS_2", "F_CIGS_3", "F_MEDUC", "F_MPCB",
    "F_M_HT", "F_PWGT", "F_RF_GDIAB", "F_RF_GHYPER", "F_RF_PDIAB", "F_RF_PHYPER",
    "F_TOBACO", "F_WIC", "IMP_PLUR", "MAGE_REPFLG", "MRACEIMP", "FAGERPT_FLG",
    "IMP_SEX", "MAGE_IMPFLG"
]

UNKNOWN_VALUE_MAP = {
    "FAGECOMB": 99,
    "PRIORLIVE": 99,
    "PRIORDEAD": 99,
    "ILLB_R": 999,
    "ILLB_R11": 99,
    "PRECARE": 99,
    "CIG_0": 99,
    "CIG_1": 99,
    "CIG_2": 99,
    "M_Ht_In": 99,
    "DLMP_MM": 99,
    "BMI": 99.9,
    "FAGEREC11": 11,
    "MEDUC": 9,
    "LBO_REC": 9,
    "PWgt_R": 999
}

# Feature Groups for Experiments
CIGARETTE_COLUMNS = ["CIG_0", "CIG_1", "CIG_2", "CIG_0_UNKNOWN", "CIG_1_UNKNOWN", "CIG_2_UNKNOWN"]
CIGARETTE_RECODE = ["CIG_REC"]

FATHER_AGE = ["FAGECOMB", "FAGECOMB_UNKNOWN"]
FATHER_AGE_RECODE = ["FAGEREC11", "FAGEREC11_UNKNOWN"]

MOTHER_AGE = ["MAGER"]
MOTHER_AGE_RECODE_SMALL = ["MAGER9"]
MOTHER_AGE_RECODE_LARGE = ["MAGER14"]

MOTHER_RACE_RECODE_SMALL = ["MRACE6"]
MOTHER_RACE_RECODE_LARGE = ["MRACE15"]

BMI_COLS = ["BMI", "BMI_UNKNOWN"]
BMI_RECODE = ["BMI_R"]

ILLB_COLS = ["ILLB_R", "ILLB_R_UNKNOWN"]
ILLB_RECODE = ["ILLB_R11", "ILLB_R11_UNKNOWN"]

# Composite Groups
NUMERIC_FEATURE_SET = CIGARETTE_COLUMNS + FATHER_AGE + MOTHER_AGE + BMI_COLS + ILLB_COLS
RECODE_COMMON = CIGARETTE_RECODE + FATHER_AGE_RECODE + BMI_RECODE + ILLB_RECODE
RECODE_SMALL_SPECIFIC = MOTHER_AGE_RECODE_SMALL + MOTHER_RACE_RECODE_SMALL
RECODE_LARGE_SPECIFIC = MOTHER_AGE_RECODE_LARGE + MOTHER_RACE_RECODE_LARGE


def build_unknown_mask(series, unknown_values):
    if not isinstance(unknown_values, (list, tuple, set)):
        unknown_values = [unknown_values]

    mask = pd.Series(False, index=series.index)

    # Numeric comparison
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_unknowns = pd.to_numeric(pd.Series(list(unknown_values)), errors="coerce").dropna().tolist()
    if numeric_unknowns:
        mask = mask | numeric_series.isin(numeric_unknowns)

    # String comparison
    string_series = series.astype(str).str.strip()
    string_unknowns = [str(v).strip() for v in unknown_values if pd.isna(pd.to_numeric(v, errors="coerce"))]
    if string_unknowns:
        mask = mask | string_series.isin(string_unknowns)

    return mask


def load_data(data_path):
    logger.info(f"Loading data from {data_path}...")
    dataframes = {}
    if not os.path.exists(data_path):
        logger.error(f"Data path {data_path} does not exist.")
        return pd.DataFrame()

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            df_key = "temp_" + "_".join(file.split("_", 2)[:2])
            file_path = os.path.join(data_path, file)
            logger.info(f"Loading {file}...")
            
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except Exception as e:
                logger.warning(f"Failed to read {file}: {e}")
                continue

            year_str = file[:4]
            df["YEAR"] = pd.to_numeric(year_str, errors="coerce")

            if "NATAL" in file.upper():
                df["OUTCOME"] = 0
            elif "FETAL" in file.upper():
                df["OUTCOME"] = 1
            else:
                df["OUTCOME"] = pd.NA
            
            dataframes[df_key] = df

    if not dataframes:
        logger.error("No dataframes loaded.")
        return pd.DataFrame()

    combined_df = pd.concat(dataframes.values(), ignore_index=True)
    logger.info(f"Combined dataframe shape: {combined_df.shape}")
    return combined_df


def clean_data(df, include_reporting_flags=True):
    logger.info("Cleaning data...")
    
    # 1. Fill reporting flags with 0s (as they are boolean)
    for col in REPORTING_FLAGS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 2. Drop rows that are almost completely blank (threshold based on full column set)
    initial_count = len(df)
    df.dropna(thresh=36, inplace=True)
    logger.info(f"Dropped {initial_count - len(df)} nearly blank rows (threshold=36).")

    # 3. Filter for Gestational Age >= 20 weeks
    if "COMBGEST" in df.columns:
        # Ensure COMBGEST is numeric, coercing errors to NaN
        df["COMBGEST"] = pd.to_numeric(df["COMBGEST"], errors="coerce")
        # Keep rows with GA >= 20
        df = df[df["COMBGEST"] >= 20].copy()
        logger.info(f"Filtered for Gestational Age >= 20. Current rows: {len(df)}")
    else:
        logger.warning("COMBGEST column not found; cannot filter by gestational age.")
    
    # 4. Drop data leak columns
    existing_leak_cols = [c for c in DATA_LEAK_COLUMNS if c in df.columns]
    if existing_leak_cols:
        df.drop(columns=existing_leak_cols, inplace=True)
        logger.info(f"Dropped {len(existing_leak_cols)} data leak columns.")

    # 5. Handle Reporting Flags (Keep or Drop)
    if not include_reporting_flags:
        # If not including, drop them
        existing_flags = [c for c in REPORTING_FLAGS if c in df.columns]
        if existing_flags:
            df.drop(columns=existing_flags, inplace=True)
            logger.info(f"Dropped {len(existing_flags)} reporting flags (experiment config).")

    # 6. Final drop for any remaining rows with nulls
    initial_len = len(df)
    df.dropna(how='any', inplace=True)
    logger.info(f"Final null drop. Rows reduced from {initial_len} to {len(df)}.")

    return df


def handle_unknowns(df):
    logger.info("Handling unknown values...")
    for col, unknown_values in UNKNOWN_VALUE_MAP.items():
        if col in df.columns:
            unknown_col_name = f"{col}_UNKNOWN"
            unknown_mask = build_unknown_mask(df[col], unknown_values)
            df[unknown_col_name] = unknown_mask.astype(bool)
            df.loc[unknown_mask, col] = np.nan
    return df


def engineer_features(df):
    logger.info("Engineering features...")

    # First Birth
    if "ILLB_R" in df.columns and "ILLB_R11" in df.columns:
        illb_r_num = pd.to_numeric(df["ILLB_R"], errors="coerce")
        illb_r11_num = pd.to_numeric(df["ILLB_R11"], errors="coerce")
        first_birth_mask = (illb_r_num == 888) | (illb_r11_num == 88)
        
        df.loc[first_birth_mask, "ILLB_R"] = np.nan
        df.loc[first_birth_mask, "ILLB_R11"] = np.nan
        df["FIRST_BIRTH"] = first_birth_mask.astype(int)

    # Zero values in ILLB
    if "ILLB" in df.columns and "ILLB_R11" in df.columns:
        illb_num = pd.to_numeric(df["ILLB"], errors="coerce")
        illb_r11_num = pd.to_numeric(df["ILLB_R11"], errors="coerce")
        
        illb_zero_mask = (illb_num.isin([0, 1, 2, 3]))
        illb_r11_zero_mask = (illb_r11_num == 0)
        
        df.loc[illb_zero_mask, "ILLB"] = np.nan
        df.loc[illb_r11_zero_mask, "ILLB_R11"] = np.nan

    # BMI and Weight calculations
    if "M_Ht_In" in df.columns and "PWgt_R" in df.columns:
        df["M_ht_M"] = pd.to_numeric(df["M_Ht_In"], errors='coerce') * 0.0254
        df["PWgt_kg"] = pd.to_numeric(df["PWgt_R"], errors='coerce') * 0.453592
        df["Pre_BMI"] = df["PWgt_kg"] / (df["M_ht_M"] ** 2)

        df.drop(columns=["M_Ht_In", "PWgt_R"], inplace=True)
    
    if "BMI" in df.columns and "Pre_BMI" in df.columns:
         df["BMI_delta"] = pd.to_numeric(df["BMI"], errors='coerce') - df["Pre_BMI"]
         df["BMI_ratio"] = df["Pre_BMI"] / pd.to_numeric(df["BMI"], errors='coerce')

    # Obesity Risk Factor
    if "BMI" in df.columns:
        bmi_numeric = pd.to_numeric(df["BMI"], errors="coerce")
        df["RF_obesity"] = np.where(bmi_numeric >= 40, "Y", "N")

    # Risk Factor Count
    risk_factor_cols = ["RF_ARTEC", "RF_EHYPE", "RF_FEDRG", "RF_GDIAB", "RF_GHYPE", "RF_INFTR", "RF_obesity"]
    available_rf_cols = [c for c in risk_factor_cols if c in df.columns]
    if available_rf_cols:
        df["RF_ct"] = (
            df[available_rf_cols]
            .apply(lambda s: s.astype(str).str.strip().str.upper().eq("Y"))
            .sum(axis=1)
            .astype(int)
        )
    
    return df


def set_dtypes(df):
    logger.info("Setting data types...")
    
    # Categoricals
    categorical_cols = [
        "BMI_R", "CIG_REC", "FAGEREC11", "MAGER14", "MAGER9", "MEDUC",
        "DLMP_MM", "MBSTATE_REC", "MRACE15", "MRACE6", "MRACEHISP", "RESTATUS"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col], ordered=True)

    # Booleans (Reporting flags)
    for col in REPORTING_FLAGS:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Memory optimization
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
            
    return df


def select_features(df, feature_set):
    logger.info(f"Selecting features for set: {feature_set}")
    
    # Define what to DROP based on what we want to KEEP
    cols_to_drop = []

    # Helper sets
    all_numeric = set(NUMERIC_FEATURE_SET)
    all_recode_common = set(RECODE_COMMON)
    all_recode_small = set(RECODE_SMALL_SPECIFIC)
    all_recode_large = set(RECODE_LARGE_SPECIFIC)
    
    # Determine what to keep
    keep_cols = set()
    
    if feature_set == "numeric":
        keep_cols = all_numeric
    elif feature_set == "recode_small":
        keep_cols = all_recode_common | all_recode_small
    elif feature_set == "recode_large":
        keep_cols = all_recode_common | all_recode_large
    elif feature_set == "both_small":
        keep_cols = all_numeric | all_recode_common | all_recode_small
    elif feature_set == "both_large":
        keep_cols = all_numeric | all_recode_common | all_recode_large
        
    # Identification of columns to drop
    # We essentially want to drop any column that IS in one of our known sets
    # but NOT in our keep_cols set.
    # We don't want to drop columns that aren't in ANY of our sets (like 'YEAR', 'OUTCOME', 'RF_ct', etc.)
    
    known_features = all_numeric | all_recode_common | all_recode_small | all_recode_large
    
    for col in df.columns:
        if col in known_features and col not in keep_cols:
            cols_to_drop.append(col)

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Dropped {len(cols_to_drop)} columns for feature set '{feature_set}'.")
        
    return df


def process_data(input_path, output_path, include_reporting_flags, feature_set):
    df = load_data(input_path)
    if df.empty:
        logger.error("Empty dataframe. Exiting.")
        return

    df = clean_data(df, include_reporting_flags=include_reporting_flags)
    df = handle_unknowns(df)
    df = engineer_features(df)
    df = set_dtypes(df)
    df = select_features(df, feature_set)

    # Convert to absolute path to avoid any ambiguity
    abs_output_path = os.path.abspath(output_path)
    logger.info(f"Saving processed data to {abs_output_path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
    
    try:
        # Use chunksize to reduce memory pressure and potentially bypass writer issues
        df.to_csv(abs_output_path, index=False, chunksize=100000, encoding='utf-8')
        logger.info("Data processing complete.")
    except OSError as e:
        logger.error(f"Failed to save data to {abs_output_path}: {e}")
        # Fallback: try saving to current directory with a simple name
        fallback_path = "final_dataset_fallback.csv"
        logger.info(f"Attempting fallback save to {fallback_path}...")
        try:
            df.to_csv(fallback_path, index=False, chunksize=100000, encoding='utf-8')
            logger.info(f"Fallback save successful: {fallback_path}")
        except Exception as e2:
             logger.error(f"Fallback save also failed: {e2}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during save: {e}")


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Build features for fetal death prediction.")
        parser.add_argument("--include-reporting-flags", action="store_true", help="Include reporting flags in the dataset.")
        parser.add_argument("--feature-set", 
                            choices=["numeric", "recode_small", "recode_large", "both_small", "both_large"], 
                            default="both_large", 
                            help="Which set of features to use.")
        parser.add_argument("--input-dir", default="data/processed", help="Directory containing aligned CSV files.")
        parser.add_argument("--output-file", default="data/processed/final_dataset.csv", help="Path to save the final dataset.")
        
        args = parser.parse_args()

    # Use getattr to safely access arguments that might not be present if called from main.py
    input_dir = getattr(args, "input_dir", "data/processed")
    output_file = getattr(args, "output_file", "data/processed/final_dataset.csv")
    
    process_data(
        input_path=input_dir,
        output_path=output_file,
        include_reporting_flags=args.include_reporting_flags,
        feature_set=args.feature_set
    )

if __name__ == "__main__":
    main()
