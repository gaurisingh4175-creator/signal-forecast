import pandas as pd

FILE_PATH = "data/File A- signal_rsrp_India_Today_(2026-07-05)_detailed.xlsx"

def load_sheet(sheet_name):
    # Read raw, no header assumed
    raw = pd.read_excel(FILE_PATH, sheet_name=sheet_name, header=None)

    # Find the row where the first cell literally says 'Timestamp'
    header_row_idx = raw[raw[0] == 'Timestamp'].index[0]

    # Use that row as column names, keep only rows after it
    new_header = raw.iloc[header_row_idx]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = new_header
    df = df.reset_index(drop=True)
    return df

# Load all four quality sheets
excellent = load_sheet("Excellent")
good = load_sheet("Good")
average = load_sheet("Average")
poor = load_sheet("Poor")

for name, df in [("Excellent", excellent), ("Good", good), ("Average", average), ("Poor", poor)]:
    print(f"\n--- '{name}' sheet ---")
    print("Row count:", len(df))
    print("Columns:", df.columns.tolist())

print("\n--- 'Good' sheet preview ---")
print(good.head())

print("\nDtypes of key columns:")
print(good[['Timestamp', 'RSRP (dBm)', 'RSRQ (dB)', 'MOS Score']].dtypes)