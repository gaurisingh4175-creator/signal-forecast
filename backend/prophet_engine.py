import time
import pandas as pd
from prophet import Prophet


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


def run_forecast_prophet(file_path, train_days=7, predict_days=7):
    """
    predict_days: how many days forward to forecast (1-10).
    train_days: accepted for API symmetry with Random Forest, but Prophet
                always trains on the actual real-day data available.
    """
    train_days = clamp_days(train_days, default=7)
    predict_days = clamp_days(predict_days, default=7)

    t_start = time.time()
    print(f"[Prophet] Config: predict_days={predict_days} (train_days={train_days} not used — Prophet fits on real data only)")

    print("[Prophet] Step 1: Loading sheets...")
    sheets = ["Excellent", "Good", "Average", "Poor"]
    all_data = pd.concat([load_sheet(file_path, s) for s in sheets], ignore_index=True)
    df = all_data[['Timestamp', 'RSRP (dBm)', 'RSRQ (dB)', 'MOS Score']].copy()
    df.columns = ['Timestamp', 'RSRP', 'RSRQ', 'MOS']

    print("[Prophet] Step 2: Cleaning data...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['RSRP'] = pd.to_numeric(df['RSRP'], errors='coerce')
    df['RSRQ'] = pd.to_numeric(df['RSRQ'], errors='coerce')
    df['MOS'] = pd.to_numeric(df['MOS'], errors='coerce')
    df = df.dropna(subset=['Timestamp', 'RSRP', 'RSRQ', 'MOS'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    print("[Prophet] Step 3: Resampling to 15-min grid...")
    base = df.set_index('Timestamp')[['RSRP', 'RSRQ', 'MOS']].resample('15min').mean()
    base = base.interpolate().reset_index()
    print(f"  {len(base)} rows after resampling")

    def fit_and_forecast(metric):
        prophet_df = base[['Timestamp', metric]].rename(columns={'Timestamp': 'ds', metric: 'y'})
        model = Prophet(
            growth='flat',
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False,
            interval_width=0.8
        )
        model.fit(prophet_df)

        n_steps = predict_days * 24 * 4
        future = model.make_future_dataframe(periods=n_steps, freq='15min')
        forecast = model.predict(future)

        last_real_ts = prophet_df['ds'].max()
        future_only = forecast[forecast['ds'] > last_real_ts][['ds', 'yhat']].copy()
        future_only.columns = ['Timestamp', metric]
        return future_only

    print("[Prophet] Step 4: Fitting RSRP model...")
    rsrp_forecast = fit_and_forecast('RSRP')

    print("[Prophet] Step 5: Fitting RSRQ model...")
    rsrq_forecast = fit_and_forecast('RSRQ')

    print("[Prophet] Step 6: Fitting MOS model...")
    mos_forecast = fit_and_forecast('MOS')

    forecast = rsrp_forecast.merge(rsrq_forecast, on='Timestamp').merge(mos_forecast, on='Timestamp')

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

    print(f"[Prophet] TOTAL TIME: {time.time() - t_start:.2f}s")

    return {
        "forecast": forecast_out.to_dict(orient='records'),
        "historical": historical_out,
        "summary": summary.to_dict(orient='records'),
        "total_points": len(forecast),
        "train_days": train_days,
        "predict_days": predict_days
    }