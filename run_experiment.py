import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_data, prepare_series_data
from src.preprocessing import SeriesScaler
from src.models import CatBoostModel
from src.metrics import calculate_all_metrics
from config import HORIZON

df = generate_synthetic_data(n_series=20, n_points=100)
series_dict = prepare_series_data(df, horizon=HORIZON)

all_results = []

for series_id, data in series_dict.items():
    train = data['train']
    test = data['test']
    
    for method in ['none', 'standard', 'robust']:
        scaler = SeriesScaler(method=method)
        scaler.fit(train)
        train_scaled = scaler.transform(train)
        
        X_train = []
        y_train = []
        lookback = 12
        
        for i in range(lookback, len(train_scaled) - HORIZON):
            X_train.append(train_scaled[i-lookback:i])
            y_train.append(train_scaled[i])
        
        if len(X_train) == 0:
            continue
        
        X_train = np.array([x.flatten() for x in X_train])
        y_train = np.array(y_train)
        
        model = CatBoostModel(horizon=HORIZON)
        model.fit(X_train, y_train)
        
        pred_scaled = []
        current_input = train_scaled[-lookback:].copy()
        
        for step in range(HORIZON):
            X_test = np.array([current_input.flatten()])
            next_pred = model.predict(X_test)[0]
            pred_scaled.append(next_pred)
            current_input = np.roll(current_input, -1)
            current_input[-1] = next_pred
        
        pred = scaler.inverse_transform(np.array(pred_scaled))
        metrics = calculate_all_metrics(test, pred, train)
        
        all_results.append({
            'series': series_id,
            'method': method,
            'sMAPE': metrics['sMAPE'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE']
        })

results_df = pd.DataFrame(all_results)
results_df.to_csv("results/experiment_results.csv", index=False)
