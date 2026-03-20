import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

N_SERIES = 200  
HORIZON = 12    
VAL_SIZE = 12   

RANDOM_STATE = 42
N_SPLITS = 3   

SCALING_METHODS = ['none', 'standard', 'robust', 'quantile']

MODELS = ['catboost', 'patchtst']

BASELINES = ['naive', 'seasonal_naive', 'auto_theta', 'auto_ets']