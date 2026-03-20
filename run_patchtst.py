import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_data, prepare_series_data
from src.preprocessing import SeriesScaler
from src.models import PatchTSTWrapper
from src.metrics import calculate_all_metrics
from config import HORIZON

df = generate_synthetic_data(n_series=10, n_points=150)
series_dict = prepare_series_data(df, horizon=HORIZON)

all_results = []

for series_id, data in series_dict.items():
    train = data['train']
    test = data['test']
    
    if len(train) < 60:
        continue
    
    for method in ['none', 'standard', 'robust']:
        scaler = SeriesScaler(method=method)
        scaler.fit(train)
        train_scaled = scaler.transform(train)
        
        model = PatchTSTWrapper(input_length=24, horizon=HORIZON, epochs=30)
        model.fit(train_scaled)
        
        pred_scaled = model.predict(train_scaled[-24:])
        pred = scaler.inverse_transform(pred_scaled)
        
        metrics = calculate_all_metrics(test, pred, train)
        all_results.append({
            'series': series_id,
            'method': method,
            'sMAPE': metrics['sMAPE'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE']
        })

results_df = pd.DataFrame(all_results)
results_df.to_csv("results/patchtst_results.csv", index=False)

print("\nPatchTST Results:")
print(results_df.groupby('method')[['sMAPE', 'MAE', 'RMSE']].mean().round(2))
