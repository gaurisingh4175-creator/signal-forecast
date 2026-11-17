import pandas as pd


def load_sheet(file_path, sheet_name):
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    header_row_idx = raw[raw[0] == 'Timestamp'].index[0]
    new_header = raw.iloc[header_row_idx]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = new_header
    df = df.reset_index(drop=True)
    return df


def get_locations(file_path, max_points=500):
    sheets = ["Excellent", "Good", "Average", "Poor"]
    all_data = pd.concat([load_sheet(file_path, s) for s in sheets], ignore_index=True)
    df = all_data[['Latitude', 'Longitude', 'MOS Score', 'MOS Grade']].copy()
    df.columns = ['Latitude', 'Longitude', 'MOS', 'Quality']

    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['MOS'] = pd.to_numeric(df['MOS'], errors='coerce')
    df = df.dropna()

    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    df['MOS'] = df['MOS'].round(2)
    return df.to_dict(orient='records')
import math


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_nearest_good_signal(file_path, user_lat, user_lon):
    """
    Given the user's current coordinates, finds the closest recorded reading
    classified as 'Good' or 'Excellent', and the closest 'Excellent' reading
    specifically, from the dataset's real GPS-tagged measurements.
    """
    sheets = ["Excellent", "Good", "Average", "Poor"]
    all_data = pd.concat([load_sheet(file_path, s) for s in sheets], ignore_index=True)
    df = all_data[['Latitude', 'Longitude', 'MOS Score', 'MOS Grade']].copy()
    df.columns = ['Latitude', 'Longitude', 'MOS', 'Quality']

    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['MOS'] = pd.to_numeric(df['MOS'], errors='coerce')
    df = df.dropna()

    df['distance_km'] = df.apply(
        lambda r: haversine_km(user_lat, user_lon, r['Latitude'], r['Longitude']), axis=1
    )

    good_or_better = df[df['Quality'].isin(['Good', 'Excellent'])]
    excellent_only = df[df['Quality'] == 'Excellent']

    def nearest_row(subset):
        if len(subset) == 0:
            return None
        row = subset.loc[subset['distance_km'].idxmin()]
        return {
            "distance_km": round(float(row['distance_km']), 2),
            "latitude": round(float(row['Latitude']), 6),
            "longitude": round(float(row['Longitude']), 6),
            "mos": round(float(row['MOS']), 2),
            "quality": row['Quality']
        }

    return {
        "nearest_good_or_better": nearest_row(good_or_better),
        "nearest_excellent": nearest_row(excellent_only),
        "total_points_checked": len(df)
    }