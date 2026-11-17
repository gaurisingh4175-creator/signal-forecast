import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def load_sheet(file_path, sheet_name):
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    header_row_idx = raw[raw[0] == 'Timestamp'].index[0]
    new_header = raw.iloc[header_row_idx]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = new_header
    df = df.reset_index(drop=True)
    return df


def clamp_days(value, default=7, lo=1, hi=10):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def run_forecast(file_path, train_days=7, predict_days=7):
    """
    train_days: how many days of synthetic history to build from the real data (1-10).
    predict_days: how many days forward to forecast (1-10).
    """
    train_days = clamp_days(train_days, default=7)
    predict_days = clamp_days(predict_days, default=7)

    t_start = time.time()
    print(f"[RF] Config: train_days={train_days}, predict_days={predict_days}")

    print("Step 1: Loading sheets...")
    sheets = ["Excellent", "Good", "Average", "Poor"]
    all_data = pd.concat([load_sheet(file_path, s) for s in sheets], ignore_index=True)
    df = all_data[['Timestamp', 'RSRP (dBm)', 'RSRQ (dB)', 'MOS Score']].copy()
    df.columns = ['Timestamp', 'RSRP', 'RSRQ', 'MOS']

    print("Step 2: Cleaning data...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['RSRP'] = pd.to_numeric(df['RSRP'], errors='coerce')
    df['RSRQ'] = pd.to_numeric(df['RSRQ'], errors='coerce')
    df['MOS'] = pd.to_numeric(df['MOS'], errors='coerce')
    df = df.dropna(subset=['Timestamp', 'RSRP', 'RSRQ', 'MOS'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    print("Step 3: Resampling to 15-min grid...")
    base = df.set_index('Timestamp')[['RSRP', 'RSRQ', 'MOS']].resample('15min').mean()
    base = base.interpolate().reset_index()
    print(f"  {len(base)} rows after resampling")

    print(f"Step 4: Building synthetic {train_days}-day history...")
    np.random.seed(42)
    synthetic_days = []
    for i in range(train_days, 0, -1):
        day = base.copy()
        day['Timestamp'] = day['Timestamp'] - pd.Timedelta(days=i)
        day['RSRP'] += np.random.normal(0, 1.5, len(day))
        day['RSRQ'] += np.random.normal(0, 0.5, len(day))
        day['MOS'] += np.random.normal(0, 0.15, len(day))
        synthetic_days.append(day)

    history = pd.concat(synthetic_days + [base], ignore_index=True)
    history = history.sort_values('Timestamp').reset_index(drop=True)
    print(f"  {len(history)} rows total ({train_days + 1} days incl. real day)")

    print("Step 5: Feature engineering...")
    history['hour'] = history['Timestamp'].dt.hour
    history['minute'] = history['Timestamp'].dt.minute
    history['dayofweek'] = history['Timestamp'].dt.dayofweek
    history['day_index'] = (history['Timestamp'] - history['Timestamp'].min()).dt.days

    LAGS = [1, 2, 3, 4, 8]
    for lag in LAGS:
        history[f'RSRP_lag{lag}'] = history['RSRP'].shift(lag)
        history[f'RSRQ_lag{lag}'] = history['RSRQ'].shift(lag)
        history[f'MOS_lag{lag}'] = history['MOS'].shift(lag)
    history = history.dropna().reset_index(drop=True)

    if len(history) < 10:
        raise ValueError(
            f"Not enough rows after feature engineering ({len(history)}). "
            f"Try increasing train_days or check the input file."
        )

    base_features = ['hour', 'minute', 'dayofweek', 'day_index']

    def train_rf(target):
        feats = base_features + [f'{target}_lag{lag}' for lag in LAGS]
        X = history[feats]
        y = history[target]
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
        model.fit(X, y)
        return model, feats

    print("Step 6: Training models...")
    t0 = time.time()
    model_rsrp, feats_rsrp = train_rf('RSRP')
    model_rsrq, feats_rsrq = train_rf('RSRQ')
    model_mos, feats_mos = train_rf('MOS')
    print(f"  done in {time.time() - t0:.2f}s")

    last_ts = history['Timestamp'].max()
    n_steps = predict_days * 24 * 4
    future_timestamps = pd.date_range(last_ts + pd.Timedelta(minutes=15), periods=n_steps, freq='15min')

    print(f"Step 7: Running recursive forecast ({n_steps} steps)...")
    t0 = time.time()

    rsrp_buf = list(history['RSRP'].to_numpy())
    rsrq_buf = list(history['RSRQ'].to_numpy())
    mos_buf = list(history['MOS'].to_numpy())
    min_ts = history['Timestamp'].min()

    preds = []
    for idx, ts in enumerate(future_timestamps):
        if idx % 100 == 0:
            print(f"  ...step {idx}/{n_steps}  ({time.time() - t0:.2f}s elapsed)")

        common = [ts.hour, ts.minute, ts.dayofweek, (ts - min_ts).days]

        rsrp_feats = common + [rsrp_buf[-lag] for lag in LAGS]
        rsrq_feats = common + [rsrq_buf[-lag] for lag in LAGS]
        mos_feats = common + [mos_buf[-lag] for lag in LAGS]

        pred_rsrp = model_rsrp.predict(np.array([rsrp_feats]))[0]
        pred_rsrq = model_rsrq.predict(np.array([rsrq_feats]))[0]
        pred_mos = model_mos.predict(np.array([mos_feats]))[0]

        preds.append({'Timestamp': ts, 'RSRP': pred_rsrp, 'RSRQ': pred_rsrq, 'MOS': pred_mos})
        rsrp_buf.append(pred_rsrp)
        rsrq_buf.append(pred_rsrq)
        mos_buf.append(pred_mos)

    forecast = pd.DataFrame(preds)
    print(f"  done in {time.time() - t0:.2f}s")

    print("Step 8: Building historical + summary output...")

    def mos_grade(mos):
        if mos >= 4.0:
            return 'Excellent'
        elif mos >= 3.5:
            return 'Good'
        elif mos >= 3.0:
            return 'Average'
        else:
            return 'Poor'

    # Historical series (the real/base data) for the "historical vs forecast" toggle
    historical = base.copy()
    historical['Quality'] = historical['MOS'].apply(mos_grade)
    historical['Timestamp'] = historical['Timestamp'].astype(str)
    historical['RSRP'] = historical['RSRP'].round(2)
    historical['RSRQ'] = historical['RSRQ'].round(2)
    historical['MOS'] = historical['MOS'].round(2)
    historical_out = historical.to_dict(orient='records')

    forecast['Quality'] = forecast['MOS'].apply(mos_grade)
    summary = forecast['Quality'].value_counts().reset_index()
    summary.columns = ['Quality', 'Count']
    summary['Percent'] = (summary['Count'] / len(forecast) * 100).round(2)

    forecast_out = forecast.copy()
    forecast_out['Timestamp'] = forecast_out['Timestamp'].astype(str)
    forecast_out['RSRP'] = forecast_out['RSRP'].round(2)
    forecast_out['RSRQ'] = forecast_out['RSRQ'].round(2)
    forecast_out['MOS'] = forecast_out['MOS'].round(2)

    print(f"TOTAL TIME: {time.time() - t_start:.2f}s")

    return {
        "forecast": forecast_out.to_dict(orient='records'),
        "historical": historical_out,
        "summary": summary.to_dict(orient='records'),
        "total_points": len(forecast),
        "train_days": train_days,
        "predict_days": predict_days
    }