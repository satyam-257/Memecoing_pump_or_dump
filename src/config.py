"""Configuration and constants for Solana Memecoins Pump or Dump."""
import os

def get_data_dir():
    candidates = [
        '/kaggle/input/competitions/memecoins-pump-or-dump',
        '/kaggle/input/memecoins-pump-or-dump',
        'data',
        '../data',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return 'data'

DATA_DIR = get_data_dir()

# Market & Token Identifiers
WSOL = "So11111111111111111111111111111111111111112"

# Financial & Market Constants
COST = 0.30          # Aggregate round-trip execution friction in SOL (fees + slippage)
TARGET_K = 5.0       # Primary take-profit multiplier hurdle
LEVELS = [3.0, 4.0, 5.0, 7.0, 10.0, 15.0]  # Multi-level take-profit grid
KILL_THRESH = 1.50   # Cascade stage-1 survival threshold
WIN = 3600           # Evaluation holding horizon (1 hour in seconds)
EMBARGO = 3600       # Embargo window before validation day start
HOLD_DAYS = 5        # Number of recent training days used to calibrate decision thresholds
TAU_DECAY = 5 * 86400 # 5-day exponential half-life for reputation recency decay

# Multi-seed ensemble configuration
SEEDS = [42, 101, 2024, 7, 777]

# Default LightGBM hyperparameters tuned for sparse microstructure signals
LGB_PARAMS = {
    'n_estimators': 350,
    'learning_rate': 0.04,
    'num_leaves': 31,
    'min_child_samples': 40,
    'subsample': 0.80,
    'subsample_freq': 1,
    'colsample_bytree': 0.70,
    'verbose': -1,
    'n_jobs': -1
}
