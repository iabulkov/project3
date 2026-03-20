import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

def mase(y_true, y_pred, y_train):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    naive_errors = np.abs(np.diff(y_train))
    naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1
    
    model_errors = np.abs(y_true - y_pred)
    
    return np.mean(model_errors) / naive_mae

def calculate_all_metrics(y_true, y_pred, y_train=None):
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'sMAPE': smape(y_true, y_pred)
    }
    
    if y_train is not None:
        metrics['MASE'] = mase(y_true, y_pred, y_train)
    
    return metrics