import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, N_SERIES

def load_m4_data():
    data_path = DATA_DIR / "synthetic_data.parquet"
    
    if data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        df = generate_synthetic_data()
        df.to_parquet(data_path)
    
    return df

def generate_synthetic_data(n_series=None, n_points=None):
    if n_series is None:
        n_series = N_SERIES
    if n_points is None:
        n_points = 100
    
    np.random.seed(42)
    data = []
    for i in range(n_series):
        trend = np.linspace(0, 5, n_points)
        seasonality = 5 * np.sin(2 * np.pi * np.arange(n_points) / 12)
        noise = np.random.normal(0, 1, n_points)
        series = 50 + trend + seasonality + noise
        
        for j, val in enumerate(series):
            data.append([f"Series_{i+1}", val])
    
    df = pd.DataFrame(data, columns=['V1', 'V2'])
    return df

def sample_series(df, n_series=None):
    if n_series is None:
        n_series = N_SERIES
    
    unique_ids = df['V1'].unique()
    np.random.seed(42)
    sampled_ids = np.random.choice(unique_ids, size=min(n_series, len(unique_ids)), replace=False)
    sampled_df = df[df['V1'].isin(sampled_ids)].copy()
    return sampled_df

def prepare_series_data(df, horizon=12):
    series_dict = {}
    
    for series_id in df['V1'].unique():
        series_values = df[df['V1'] == series_id]['V2'].values
        
        if len(series_values) < horizon + 10:
            continue
            
        train = series_values[:-horizon]
        test = series_values[-horizon:]
        
        series_dict[series_id] = {
            'train': train,
            'test': test,
            'full': series_values
        }
    
    return series_dict