import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os

FILE_PATH = "data/File A- signal_rsrp_India_Today_(2026-07-05)_detailed.xlsx"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Step 1: Load & combine all four quality sheets ----------
def load_sheet(sheet_name):
    raw = pd.read_excel(FILE_PATH, sheet_name=sheet_name, header=None)
    header_row_idx = raw[raw[0] == 'Timestamp'].index[0]
    new_header = raw.iloc[header_row_idx]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = new_header
    df = df.reset_index(drop=True)
    return df

sheets = ["Excellent", "Good", "Average", "Poor"]
all_data = pd.concat([load_sheet(s) for s in sheets], ignore_index=True)

# Keep only the columns we need
df = all_data[['Timestamp', 'RSRP (dBm)', 'RSRQ (dB)', 'MOS Score']].copy()
df.columns = ['Timestamp', 'RSRP', 'RSRQ', 'MOS']

# ---------- Step 2: Fix data types ----------
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['RSRP'] = pd.to_numeric(df['RSRP'], errors='coerce')
df['RSRQ'] = pd.to_numeric(df['RSRQ'], errors='coerce')
df['MOS'] = pd.to_numeric(df['MOS'], errors='coerce')

df = df.dropna(subset=['Timestamp', 'RSRP', 'RSRQ', 'MOS'])
df = df.sort_values('Timestamp').reset_index(drop=True)

print(f"Total clean rows: {len(df)}")
print(f"Time range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

# ---------- Step 3: Resample to a regular 1-minute grid ----------
base = df.set_index('Timestamp')[['RSRP', 'RSRQ', 'MOS']].resample('1min').mean()
base = base.interpolate()  # fill any small gaps
base = base.reset_index()

print(f"\nResampled to {len(base)} 1-minute rows for a single day.")

# ---------- Step 4: Build synthetic 7-day HISTORY (days -7 to -1) ----------
np.random.seed(42)
synthetic_days = []
for i in range(7, 0, -1):
    day = base.copy()
    day['Timestamp'] = day['Timestamp'] - pd.Timedelta(days=i)
    day['RSRP'] += np.random.normal(0, 1.5, len(day))
    day['RSRQ'] += np.random.normal(0, 0.5, len(day))
    day['MOS']  += np.random.normal(0, 0.15, len(day))
    synthetic_days.append(day)

history = pd.concat(synthetic_days + [base], ignore_index=True)
history = history.sort_values('Timestamp').reset_index(drop=True)

print(f"Synthetic history length (8 days total): {len(history)} rows")
history.to_csv(f"{OUTPUT_DIR}/synthetic_history.csv", index=False)

# ---------- Step 5: Feature engineering ----------
history['hour'] = history['Timestamp'].dt.hour
history['minute'] = history['Timestamp'].dt.minute
history['dayofweek'] = history['Timestamp'].dt.dayofweek
history['day_index'] = (history['Timestamp'] - history['Timestamp'].min()).dt.days

LAGS = [1, 2, 3, 15, 60]  # in minutes, since our grid is 1-min
for lag in LAGS:
    history[f'RSRP_lag{lag}'] = history['RSRP'].shift(lag)
    history[f'RSRQ_lag{lag}'] = history['RSRQ'].shift(lag)
    history[f'MOS_lag{lag}']  = history['MOS'].shift(lag)

history = history.dropna().reset_index(drop=True)

# ---------- Step 6: Train one Random Forest per metric ----------
base_features = ['hour', 'minute', 'dayofweek', 'day_index']

def train_rf(target):
    feats = base_features + [f'{target}_lag{lag}' for lag in LAGS]
    X = history[feats]
    y = history[target]
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X, y)
    return model, feats

model_rsrp, feats_rsrp = train_rf('RSRP')
model_rsrq, feats_rsrq = train_rf('RSRQ')
model_mos,  feats_mos  = train_rf('MOS')

print("\nModels trained successfully.")

# ---------- Step 7: Recursive 7-day forecast ----------
last_ts = history['Timestamp'].max()
future_timestamps = pd.date_range(last_ts + pd.Timedelta(minutes=1), periods=7*24*60, freq='1min')

recent = history[['Timestamp', 'RSRP', 'RSRQ', 'MOS']].copy()
preds = []

for ts in future_timestamps:
    row_common = {
        'hour': ts.hour,
        'minute': ts.minute,
        'dayofweek': ts.dayofweek,
        'day_index': (ts - history['Timestamp'].min()).days,
    }

    def build_row(target, feats):
        row = dict(row_common)
        for lag in LAGS:
            row[f'{target}_lag{lag}'] = recent[target].iloc[-lag]
        return pd.DataFrame([row])[feats]

    pred_rsrp = model_rsrp.predict(build_row('RSRP', feats_rsrp))[0]
    pred_rsrq = model_rsrq.predict(build_row('RSRQ', feats_rsrq))[0]
    pred_mos  = model_mos.predict(build_row('MOS', feats_mos))[0]

    preds.append({'Timestamp': ts, 'RSRP': pred_rsrp, 'RSRQ': pred_rsrq, 'MOS': pred_mos})
    recent = pd.concat([recent, pd.DataFrame([preds[-1]])], ignore_index=True)

forecast = pd.DataFrame(preds)
forecast.to_csv(f"{OUTPUT_DIR}/forecast_7day.csv", index=False)
print(f"\n7-day forecast saved: {OUTPUT_DIR}/forecast_7day.csv ({len(forecast)} rows)")

# ---------- Step 8: Re-bucket into quality grades (based on MOS) ----------
def mos_grade(mos):
    if mos >= 4.0: return 'Excellent'
    elif mos >= 3.5: return 'Good'
    elif mos >= 3.0: return 'Average'
    else: return 'Poor'

forecast['Quality'] = forecast['MOS'].apply(mos_grade)
summary_forecast = forecast['Quality'].value_counts().reset_index()
summary_forecast.columns = ['Quality', 'Count']
summary_forecast['% of Total'] = (summary_forecast['Count'] / len(forecast) * 100).round(2)
summary_forecast.to_csv(f"{OUTPUT_DIR}/forecast_summary.csv", index=False)

print("\n--- Forecast Quality Summary ---")
print(summary_forecast)

# ---------- Step 9: Plots ----------
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(forecast['Timestamp'], forecast['RSRP'])
axes[0].set_title('7-Day RSRP Forecast (dBm)')
axes[1].plot(forecast['Timestamp'], forecast['RSRQ'], color='orange')
axes[1].set_title('7-Day RSRQ Forecast (dB)')
axes[2].plot(forecast['Timestamp'], forecast['MOS'], color='green')
axes[2].set_title('7-Day MOS Forecast')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/forecast_plots.png")
print(f"\nPlots saved: {OUTPUT_DIR}/forecast_plots.png")