import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class SeriesScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method='standard'):
        self.method = method
        self.scaler = None
        
    def fit(self, X, y=None):
        X = np.array(X).reshape(-1, 1)
        
        if self.method == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        elif self.method == 'robust':
            self.scaler = RobustScaler()
            self.scaler.fit(X)
        elif self.method == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal')
            self.scaler.fit(X)
        elif self.method == 'none':
            pass
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        return self
    
    def transform(self, X):
        X = np.array(X).reshape(-1, 1)
        
        if self.method == 'none':
            return X.flatten()
        
        return self.scaler.transform(X).flatten()
    
    def inverse_transform(self, X):
        X = np.array(X).reshape(-1, 1)
        
        if self.method == 'none':
            return X.flatten()
        
        return self.scaler.inverse_transform(X).flatten()

def apply_scaling_to_series(series_dict, scaling_methods):
    scaled_series = {}
    
    for series_id, data in series_dict.items():
        train_data = data['train']
        test_data = data['test']
        
        scaled_series[series_id] = {}
        
        for method in scaling_methods:
            scaler = SeriesScaler(method=method)
            scaler.fit(train_data)
            
            scaled_train = scaler.transform(train_data)
            scaled_test = scaler.transform(test_data)
            
            scaled_series[series_id][method] = {
                'train': scaled_train,
                'test': scaled_test,
                'scaler': scaler,
                'original_train': train_data,
                'original_test': test_data
            }
    
    return scaled_series