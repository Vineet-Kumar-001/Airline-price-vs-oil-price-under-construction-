import pandas as pd
import numpy as np
import logging
import joblib
import os
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Optimization - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_next_model_version(model_dir: Path) -> int:
    existing_models = list(model_dir.glob("model_v*.pkl"))
    if not existing_models:
        return 1
    versions = []
    for model_path in existing_models:
        match = re.search(r"model_v(\d+)\.pkl", model_path.name)
        if match:
            versions.append(int(match.group(1)))
    return max(versions) + 1 if versions else 1

def update_model_registry(registry_path: Path, version: int, rmse: float, mae: float, r2: float, params: dict, model_path: Path):
    """
    Appends the new model's metadata to the CSV Model Registry.
    """
    # Convert parameters dictionary to a JSON string so it fits in one CSV column
    params_str = json.dumps(params)
    
    new_entry = pd.DataFrame([{
        'version': f"v{version}",
        'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'r2_score': round(r2, 4),
        'status': 'Staging', # All new models start in Staging
        'model_path': str(model_path.relative_to(registry_path.parent.parent)),
        'hyperparameters': params_str
    }])

    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        registry = pd.concat([registry, new_entry], ignore_index=True)
    else:
        registry = new_entry
        
    registry.to_csv(registry_path, index=False)
    logger.info(f"📝 Logged Model v{version} to Model Registry.")

def optimize_and_save_model():
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent 
    
    input_data_path = PROJECT_ROOT / "data" / "processed_data" / "model_ready_data.csv"
    
    model_output_dir = PROJECT_ROOT / "models" / "model_versions"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    graph_output_dir = PROJECT_ROOT / "models" / "model_graph"
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Path for the Model Registry
    registry_path = PROJECT_ROOT / "models" / "model_registry.csv"
    
    next_version = get_next_model_version(model_output_dir)
    logger.info(f"Preparing to train and save Model Version: v{next_version}")
    
    df = pd.read_csv(input_data_path)
    
    target_col = 'net_profit_usd_m'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 5, 6, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    xgb_estimator = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_estimator,
        param_distributions=param_distributions,
        n_iter=10,          
        scoring='neg_mean_squared_error',
        cv=3,               
        verbose=1,
        random_state=42,  # <--- Change this to None
        n_jobs=-1           
    )
    
    logger.info("Running RandomizedSearchCV...")
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"📈 OPTIMIZED Model Metrics (v{next_version}): RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    
    # Save the model
    model_filename = f"model_v{next_version}.pkl"
    model_path = model_output_dir / model_filename
    joblib.dump(best_model, model_path)
    
    # LOG TO REGISTRY
    update_model_registry(registry_path, next_version, rmse, mae, r2, best_params, model_path)
    
    # Save Plot
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 8))
    importance = pd.Series(best_model.feature_importances_, index=X.columns)
    importance.nlargest(15).sort_values(ascending=True).plot(kind='barh', color='#00ff00')
    plt.title(f"Top 15 Feature Importances (v{next_version})")
    plt.tight_layout()
    plt.savefig(graph_output_dir / f"feature_importance_v{next_version}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    optimize_and_save_model()