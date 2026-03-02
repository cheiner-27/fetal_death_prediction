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
CIGARETTE_COLUMNS = ["CIG_0", "CIG_1", "CIG_2"]
CIGARETTE_RECODE = ["CIG_REC"]

FATHER_AGE = ["FAGECOMB"]
FATHER_AGE_RECODE = ["FAGEREC11"]

MOTHER_AGE = ["MAGER"]
MOTHER_AGE_RECODE_SMALL = ["MAGER9"]
MOTHER_AGE_RECODE_LARGE = ["MAGER14"]

MOTHER_RACE_RECODE_SMALL = ["MRACE6"]
MOTHER_RACE_RECODE_LARGE = ["MRACE15"]

BMI_COLS = ["BMI"]
BMI_RECODE = ["BMI_R"]

ILLB_COLS = ["ILLB_R"]
ILLB_RECODE = ["ILLB_R11"]

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


def impute_data(df):
    logger.info("Imputing missing/unknown values...")
    
    # 1. Map unknowns to NaN for identified columns to facilitate imputation
    for col, unknown_values in UNKNOWN_VALUE_MAP.items():
        if col in df.columns:
            mask = build_unknown_mask(df[col], unknown_values)
            df.loc[mask, col] = np.nan
            # Ensure numeric for calculations
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure MAGER9 exists for stratification
    if "MAGER9" not in df.columns and "MAGER" in df.columns:
        bins = [0, 15, 20, 25, 30, 35, 40, 45, 50, 150]
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        df["MAGER9"] = pd.cut(df["MAGER"], bins=bins, labels=labels, right=False).astype(float)
    elif "MAGER9" in df.columns:
        df["MAGER9"] = pd.to_numeric(df["MAGER9"], errors="coerce")

    # --- Stratified Imputation Logic ---

    # BMI: Median stratified by age (MAGER9)
    if "BMI" in df.columns:
        df["BMI"] = df.groupby("MAGER9")["BMI"].transform(lambda x: x.fillna(x.median()))

    # CIG_0: Random based on distribution of age group
    if "CIG_0" in df.columns:
        def stochastic_impute_group(group):
            if group.isnull().all(): return group
            dist = group.dropna().value_counts(normalize=True)
            if dist.empty: return group
            nans = group.isnull()
            if nans.any():
                group.loc[nans] = np.random.choice(dist.index, size=nans.sum(), p=dist.values)
            return group
        df["CIG_0"] = df.groupby("MAGER9")["CIG_0"].transform(stochastic_impute_group)

    # CIG_1, CIG_2
    if "CIG_0" in df.columns:
        if "CIG_1" in df.columns:
            df.loc[df["CIG_0"] == 0, "CIG_1"] = 0
            delta = (df["CIG_1"] - df["CIG_0"]).median()
            if pd.isna(delta): delta = 0
            df["CIG_1"] = df["CIG_1"].fillna((df["CIG_0"] + delta).clip(lower=0))
        
        if "CIG_2" in df.columns:
            df.loc[df["CIG_0"] == 0, "CIG_2"] = 0
            # Use median delta between CIG_1 and CIG_0 as requested
            delta_val = (df["CIG_1"] - df["CIG_0"]).median()
            if pd.isna(delta_val): delta_val = 0
            df["CIG_2"] = df["CIG_2"].fillna((df["CIG_0"] + delta_val).clip(lower=0))

    # DLMP_MM: Random uniform 1-12
    if "DLMP_MM" in df.columns:
        nans = df["DLMP_MM"].isnull()
        if nans.any():
            df.loc[nans, "DLMP_MM"] = np.random.randint(1, 13, size=nans.sum())

    # FAGECOMB: Median delta with mother's age
    if "FAGECOMB" in df.columns and "MAGER" in df.columns:
        delta = (df["FAGECOMB"] - df["MAGER"]).median()
        if pd.isna(delta): delta = 0
        df["FAGECOMB"] = df["FAGECOMB"].fillna(df["MAGER"] + delta)

    # FAGEREC11: Based on FAGECOMB (5-year increments)
    if "FAGEREC11" in df.columns and "FAGECOMB" in df.columns:
        def recode_fage11(age):
            if pd.isna(age): return np.nan
            if age < 15: return 1
            if age < 20: return 2
            if age < 25: return 3
            if age < 30: return 4
            if age < 35: return 5
            if age < 40: return 6
            if age < 45: return 7
            if age < 50: return 8
            if age < 55: return 9
            return 10
        nans = df["FAGEREC11"].isnull()
        if nans.any():
            df.loc[nans, "FAGEREC11"] = df.loc[nans, "FAGECOMB"].apply(recode_fage11)

    # ILLB: 888 -> 0; 999 -> median by age
    illb_col = "ILLB" if "ILLB" in df.columns else ("ILLB_R" if "ILLB_R" in df.columns else None)
    if illb_col:
        df.loc[df[illb_col] == 888, illb_col] = 0
        df[illb_col] = df.groupby("MAGER9")[illb_col].transform(lambda x: x.fillna(x.median()))

    # ILLB_R11: Recode from ILLB
    if "ILLB_R11" in df.columns and illb_col:
        def recode_illb11(val):
            if pd.isna(val): return np.nan
            if val < 4: return 0 # 000-003: Leave as is (recode to 0 for R11)
            if 4 <= val <= 11: return 1
            if 12 <= val <= 17: return 2
            if 18 <= val <= 23: return 3
            if 24 <= val <= 35: return 4
            if 36 <= val <= 47: return 5
            if 48 <= val <= 59: return 6
            if 60 <= val <= 71: return 7
            return 8 # 72mo+
        nans = df["ILLB_R11"].isnull()
        if nans.any():
            df.loc[nans, "ILLB_R11"] = df.loc[nans, illb_col].apply(recode_illb11)

    # MEDUC: <14: 1; 14-17: 2; missing and >17: Stochastic
    if "MEDUC" in df.columns and "MAGER" in df.columns:
        df.loc[df["MAGER"] < 14, "MEDUC"] = 1
        df.loc[(df["MAGER"] >= 14) & (df["MAGER"] <= 17), "MEDUC"] = 2
        
        mask = df["MEDUC"].isnull() & (df["MAGER"] > 17)
        if mask.any():
            def stochastic_meduc(group):
                if group.isnull().all(): return group
                dist = group.dropna().value_counts(normalize=True)
                if dist.empty: return group
                nans = group.isnull()
                if nans.any():
                    group.loc[nans] = np.random.choice(dist.index, size=nans.sum(), p=dist.values)
                return group
            # Apply only to the missing ones > 17
            imputed_meduc = df.groupby("MAGER9")["MEDUC"].transform(stochastic_meduc)
            df.loc[mask, "MEDUC"] = imputed_meduc.loc[mask]

    # M_Ht_In: Median stratified by age and BMI
    if "M_Ht_In" in df.columns:
        if "BMI" in df.columns:
            bmi_bin = pd.qcut(df["BMI"], 5, labels=False, duplicates='drop')
            df["M_Ht_In"] = df.groupby(["MAGER9", bmi_bin])["M_Ht_In"].transform(lambda x: x.fillna(x.median()))
        else:
            df["M_Ht_In"] = df.groupby("MAGER9")["M_Ht_In"].transform(lambda x: x.fillna(x.median()))

    # PRIORDEAD: Constant 0
    if "PRIORDEAD" in df.columns:
        df["PRIORDEAD"] = df["PRIORDEAD"].fillna(0)

    # PRIORLIVE: If ILLB is 888 (0): 0. Else median by age.
    if "PRIORLIVE" in df.columns:
        if illb_col:
            df.loc[df[illb_col] == 0, "PRIORLIVE"] = 0
        df["PRIORLIVE"] = df.groupby("MAGER9")["PRIORLIVE"].transform(lambda x: x.fillna(x.median()))

    # LBO_REC: PRIORLIVE + 1
    if "LBO_REC" in df.columns and "PRIORLIVE" in df.columns:
         df["LBO_REC"] = df["PRIORLIVE"] + 1

    # PRECARE: Mode stratified by MEDUC and MAGER9
    if "PRECARE" in df.columns:
        def fill_mode(x):
            if x.isnull().all(): return x
            m = x.mode()
            if m.empty: return x
            return x.fillna(m[0])
        df["PRECARE"] = df.groupby(["MEDUC", "MAGER9"])["PRECARE"].transform(fill_mode)

    # PWgt_R: Reverse engineer from BMI
    if "PWgt_R" in df.columns:
        if "BMI" in df.columns and "M_Ht_In" in df.columns:
            # Formula: Weight = (BMI * Height^2) / 703
            bmi_based_weight = (df["BMI"] * (df["M_Ht_In"]**2)) / 703
            # Use BMI-based weight to fill missing PWgt_R
            df["PWgt_R"] = df["PWgt_R"].fillna(bmi_based_weight)
            
        # If still missing: Median by age
        df["PWgt_R"] = df.groupby("MAGER9")["PWgt_R"].transform(lambda x: x.fillna(x.median()))

    return df


def engineer_features(df):
    logger.info("Engineering features...")

    # First Birth indicator (based on imputed ILLB)
    illb_col = "ILLB" if "ILLB" in df.columns else ("ILLB_R" if "ILLB_R" in df.columns else None)
    if illb_col:
        df["FIRST_BIRTH"] = (df[illb_col] == 0).astype(int)

    # BMI and Weight calculations
    if "M_Ht_In" in df.columns and "PWgt_R" in df.columns:
        df["M_ht_M"] = pd.to_numeric(df["M_Ht_In"], errors='coerce') * 0.0254
        df["PWgt_kg"] = pd.to_numeric(df["PWgt_R"], errors='coerce') * 0.453592
        df["Pre_BMI"] = df["PWgt_kg"] / (df["M_ht_M"] ** 2)

        # We keep M_Ht_In and PWgt_R for now as they might be in select_features
    
    if "BMI" in df.columns and "Pre_BMI" in df.columns:
         df["BMI_delta"] = pd.to_numeric(df["BMI"], errors='coerce') - df["Pre_BMI"]
         df["BMI_ratio"] = df["Pre_BMI"] / pd.to_numeric(df["BMI"], errors='coerce')

    # Obesity Risk Factor
    if "BMI" in df.columns:
        bmi_numeric = pd.to_numeric(df["BMI"], errors="coerce")
        df["RF_obesity"] = np.where(bmi_numeric >= 40, "Y", "N")

    # New Requested Features
    if "LBO_REC" in df.columns and "MAGER" in df.columns:
        # Frequency of births: LBO_REC / (MAGER - 11)
        age_diff = df["MAGER"] - 11
        df["BIRTH_FREQ"] = df["LBO_REC"] / age_diff.replace(0, np.nan)
        
        # Teenager with multiple births: MAGER < 20 & LBO_REC > 1
        df["TEEN_MULT_BIRTH"] = ((df["MAGER"] < 20) & (df["LBO_REC"] > 1)).astype(int)

    if "PRIORDEAD" in df.columns and "PRIORLIVE" in df.columns:
        # Prior mortality rate: PRIORDEAD / (PRIORLIVE + PRIORDEAD)
        total_prior = df["PRIORLIVE"] + df["PRIORDEAD"]
        df["PRIOR_MORT_RATE"] = np.where(total_prior > 0, df["PRIORDEAD"] / total_prior, 0)
        
        # History of loss: PRIORDEAD > 0
        df["HIST_LOSS"] = (df["PRIORDEAD"] > 0).astype(int)

    if "PRECARE" in df.columns:
        # Delayed care: PRECARE > 3
        df["DELAYED_CARE"] = (df["PRECARE"] > 3).astype(int)

    if "FAGECOMB" in df.columns and "MAGER" in df.columns:
        # Parental age discrepancy: FAGECOMB - MAGER
        df["AGE_DISC"] = df["FAGECOMB"] - df["MAGER"]

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
    df = impute_data(df)
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
