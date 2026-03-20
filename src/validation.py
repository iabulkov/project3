import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class TimeSeriesValidator:
    def __init__(self, n_splits=3, test_size=12):
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    def split(self, data):
        return self.tscv.split(data)
    
    def rolling_window_validation(self, series, model, horizon=12):
        predictions = []
        actuals = []
        
        for i in range(horizon, len(series) - horizon + 1, horizon):
            train = series[:i]
            val = series[i:i + horizon]
            
            model.fit(train, None)
            
            pred = model.predict(train[-24:]) if hasattr(model, 'input_length') else model.predict(train)
            
            predictions.extend(pred[:len(val)])
            actuals.extend(val)
        
        return np.array(predictions), np.array(actuals)