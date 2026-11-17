import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet


def load_sheet(file_path, sheet_name):
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    header_row_idx = raw[raw[0] == 'Timestamp'].index[0]
    new_header = raw.iloc[header_row_idx]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = new_header
    df = df.reset_index(drop=True)
    return df


def load_and_clean(file_path):
    sheets = ["Excellent", "Good", "Average", "Poor"]
    all_data = pd.concat([load_sheet(file_path, s) for s in sheets], ignore_index=True)
    df = all_data[['Timestamp', 'RSRP (dBm)', 'RSRQ (dB)', 'MOS Score']].copy()
    df.columns = ['Timestamp', 'RSRP', 'RSRQ', 'MOS']

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['RSRP'] = pd.to_numeric(df['RSRP'], errors='coerce')
    df['RSRQ'] = pd.to_numeric(df['RSRQ'], errors='coerce')
    df['MOS'] = pd.to_numeric(df['MOS'], errors='coerce')
    df = df.dropna(subset=['Timestamp', 'RSRP', 'RSRQ', 'MOS'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df


def validate_prophet(train, test, metric):
    prophet_df = train[['Timestamp', metric]].rename(columns={'Timestamp': 'ds', metric: 'y'})
    model = Prophet(growth='flat', daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
    model.fit(prophet_df)

    future = pd.DataFrame({'ds': test['Timestamp']})
    forecast = model.predict(future)

    y_true = test[metric].values
    y_pred = forecast['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def validate_random_forest(train, test, metric, lags=(1, 2, 3, 4, 8)):
    combined = pd.concat([train, test], ignore_index=True).sort_values('Timestamp').reset_index(drop=True)
    combined['hour'] = combined['Timestamp'].dt.hour
    combined['minute'] = combined['Timestamp'].dt.minute
    combined['dayofweek'] = combined['Timestamp'].dt.dayofweek

    for lag in lags:
        combined[f'{metric}_lag{lag}'] = combined[metric].shift(lag)

    combined = combined.dropna().reset_index(drop=True)

    split_ts = train['Timestamp'].max()
    train_feats = combined[combined['Timestamp'] <= split_ts]
    test_feats = combined[combined['Timestamp'] > split_ts]

    if len(test_feats) == 0:
        return None, None

    feats = ['hour', 'minute', 'dayofweek'] + [f'{metric}_lag{lag}' for lag in lags]
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    model.fit(train_feats[feats], train_feats[metric])

    y_true = test_feats[metric].values
    y_pred = model.predict(test_feats[feats])
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def run_validation(file_path, test_fraction=0.2):
    df = load_and_clean(file_path)
    base = df.set_index('Timestamp')[['RSRP', 'RSRQ', 'MOS']].resample('15min').mean()
    base = base.interpolate().reset_index()

    split_idx = int(len(base) * (1 - test_fraction))
    train = base.iloc[:split_idx].reset_index(drop=True)
    test = base.iloc[split_idx:].reset_index(drop=True)

    if len(test) < 2 or len(train) < 5:
        raise ValueError(
            f"Not enough data to validate (train={len(train)}, test={len(test)} rows). "
            f"Need a longer real-data window."
        )

    results = {}
    for metric in ['RSRP', 'RSRQ', 'MOS']:
        p_mae, p_rmse = validate_prophet(train, test, metric)
        rf_mae, rf_rmse = validate_random_forest(train, test, metric)

        results[metric] = {
            "prophet": {"mae": round(p_mae, 3), "rmse": round(p_rmse, 3)},
            "random_forest": {
                "mae": round(rf_mae, 3) if rf_mae is not None else None,
                "rmse": round(rf_rmse, 3) if rf_mae is not None else None
            }
        }

    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "results": results
    }