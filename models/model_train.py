import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BaselineTraining - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_baseline_xgboost():
    # ---------------------------------------------------------
    # 1. Setup Paths
    # ---------------------------------------------------------
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent 
    
    input_data_path = PROJECT_ROOT / "data" / "processed_data" / "model_ready_data.csv"
    
    logger.info(f"Loading processed data from {input_data_path}")
    
    if not input_data_path.exists():
        logger.error("❌ Processed data not found.")
        return

    df = pd.read_csv(input_data_path)
    
    # ---------------------------------------------------------
    # 2. Prepare Data
    # ---------------------------------------------------------
    target_col = 'net_profit_usd_m'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Data split successful. Training samples: {len(X_train)}")
    
    # ---------------------------------------------------------
    # 3. Train Baseline XGBoost
    # ---------------------------------------------------------
    logger.info("Initializing BASELINE XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    logger.info("Training baseline model...")
    xgb_model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 4. Evaluate Baseline Model
    # ---------------------------------------------------------
    logger.info("Evaluating baseline model on test data...")
    y_pred = xgb_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"📊 BASELINE Evaluation Metrics:")
    logger.info(f"   - RMSE: {rmse:.2f}")
    logger.info(f"   - MAE:  {mae:.2f}")
    logger.info(f"   - R²:   {r2:.4f}")
    logger.info("Baseline evaluation complete. Proceeding to optimization step next.")

if __name__ == "__main__":
    train_baseline_xgboost()