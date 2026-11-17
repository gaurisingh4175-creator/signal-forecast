
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# 1. FILE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_PATH = DATA_DIR / "Synthetic_Network_Data.xlsx"
OUTPUT_PATH = DATA_DIR / "engineered_signal_data.csv"


# =====================================================
# 2. LOAD DATA
# =====================================================

def load_data():

    print("\nLoading Excel dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Excel file not found at: {DATA_PATH}"
        )

    df = pd.read_excel(DATA_PATH, sheet_name="All_Data")

    print("Dataset loaded successfully!")
    print("Original shape:", df.shape)

    return df


# =====================================================
# 3. DATA CLEANING
# =====================================================

def clean_data(df):

    print("\nCleaning dataset...")

    # Remove duplicate rows
    df = df.drop_duplicates().copy()

    # Convert timestamp
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    # Remove rows with invalid timestamps
    df = df.dropna(subset=["Timestamp"])

    # Sort separately by site and time
    df = df.sort_values(
        by=["Site_ID", "Timestamp"]
    ).reset_index(drop=True)

    # Convert numerical columns
    numeric_columns = [
        "RSRP",
        "RSRQ",
        "MOS"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    print("Cleaning completed!")
    print("Cleaned shape:", df.shape)

    return df


# =====================================================
# 4. TIME-BASED FEATURE ENGINEERING
# =====================================================

def create_time_features(df):

    print("\nCreating time-based features...")

    df["Hour"] = df["Timestamp"].dt.hour
    df["Minute"] = df["Timestamp"].dt.minute
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    return df


# =====================================================
# 5. LAG FEATURES
# =====================================================

def create_lag_features(df):

    print("\nCreating lag features...")

    signal_columns = [
        "RSRP",
        "RSRQ",
        "MOS"
    ]

    for col in signal_columns:

        for lag in range(1, 5):

            df[f"{col}_Lag_{lag}"] = (
                df.groupby("Site_ID")[col]
                .shift(lag)
            )

    return df


# =====================================================
# 6. ROLLING STATISTICS
# =====================================================

def create_rolling_features(df):

    print("\nCreating rolling mean and standard deviation...")

    signal_columns = [
        "RSRP",
        "RSRQ",
        "MOS"
    ]

    for col in signal_columns:

        grouped = df.groupby("Site_ID")[col]

        # Shift first to avoid using current/future values
        previous_values = grouped.shift(1)

        df[f"{col}_Rolling_Mean"] = (
            previous_values.groupby(df["Site_ID"])
            .rolling(window=4)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"{col}_Rolling_Std"] = (
            previous_values.groupby(df["Site_ID"])
            .rolling(window=4)
            .std()
            .reset_index(level=0, drop=True)
        )

    return df


# =====================================================
# 7. SIGNAL CHANGE FEATURES
# =====================================================

def create_change_features(df):

    print("\nCreating signal change features...")

    signal_columns = [
        "RSRP",
        "RSRQ",
        "MOS"
    ]

    for col in signal_columns:

        df[f"{col}_Change"] = (
            df.groupby("Site_ID")[col]
            .diff()
        )

    return df


# =====================================================
# 8. FUTURE TARGET GENERATION
# =====================================================

def create_future_targets(df):

    print("\nCreating future prediction targets...")

    # Predict the next observation (15 minutes ahead
    # only if the data has regular 15-minute intervals).

    df["Future_RSRP"] = (
        df.groupby("Site_ID")["RSRP"]
        .shift(-1)
    )

    df["Future_Quality"] = (
        df.groupby("Site_ID")["Signal_Quality"]
        .shift(-1)
    )

    return df


# =====================================================
# 9. REMOVE INVALID ROWS
# =====================================================

def remove_invalid_rows(df):

    print("\nRemoving rows with missing features or targets...")

    required_columns = [
        "RSRP",
        "RSRQ",
        "MOS",
        "Future_RSRP",
        "Future_Quality",

        "RSRP_Lag_1",
        "RSRP_Lag_2",
        "RSRP_Lag_3",
        "RSRP_Lag_4",

        "RSRQ_Lag_1",
        "RSRQ_Lag_2",
        "RSRQ_Lag_3",
        "RSRQ_Lag_4",

        "MOS_Lag_1",
        "MOS_Lag_2",
        "MOS_Lag_3",
        "MOS_Lag_4",

        "RSRP_Rolling_Mean",
        "RSRP_Rolling_Std",

        "RSRQ_Rolling_Mean",
        "RSRQ_Rolling_Std",

        "MOS_Rolling_Mean",
        "MOS_Rolling_Std",

        "RSRP_Change",
        "RSRQ_Change",
        "MOS_Change"
    ]

    df = df.dropna(
        subset=required_columns
    ).copy()

    print("Invalid rows removed.")
    print("Final shape:", df.shape)

    return df


# =====================================================
# 10. SAVE ENGINEERED DATASET
# =====================================================

def save_data(df):

    print("\nSaving engineered dataset...")

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nDataset saved successfully!")
    print("Saved location:", OUTPUT_PATH)


# =====================================================
# 11. MAIN FUNCTION
# =====================================================

def main():

    print("=" * 55)
    print("MOBILE NETWORK FEATURE ENGINEERING")
    print("=" * 55)

    df = load_data()

    df = clean_data(df)

    df = create_time_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_change_features(df)

    df = create_future_targets(df)

    df = remove_invalid_rows(df)

    save_data(df)

    print("\n" + "=" * 55)
    print("FEATURE ENGINEERING COMPLETED!")
    print("=" * 55)

    print("\nFinal dataset preview:")
    print(df.head())

    print("\nSignal quality distribution:")
    print(df["Future_Quality"].value_counts())


if __name__ == "__main__":
    main()
