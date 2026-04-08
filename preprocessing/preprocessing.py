import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Preprocessing")

# ---------------------------------------------------------
# Custom Transformers for Production Pipeline
# ---------------------------------------------------------
class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Drops unnecessary columns and ensures correct data types.
    """
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else ['quarter', 'month']
        
    def fit(self, X, y=None):
        return self # Nothing to fit

    def transform(self, X):
        X_copy = X.copy()
        logger.info(f"Initial shape: {X_copy.shape}")
        
        # Drop columns
        existing_cols_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]
        if existing_cols_to_drop:
            X_copy = X_copy.drop(columns=existing_cols_to_drop)
            logger.info(f"Dropped columns: {existing_cols_to_drop}")
            
        return X_copy

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Clips extreme outliers using the IQR method to prevent model skewing.
    We only apply this to highly skewed continuous features (found during EDA).
    """
    def __init__(self, factor=1.5, columns=None):
        self.factor = factor
        self.columns = columns
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        # Calculate bounds only on training data
        cols_to_process = self.columns if self.columns else X.select_dtypes(include=[np.number]).columns
        
        for col in cols_to_process:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - (self.factor * iqr)
            self.upper_bounds_[col] = q3 + (self.factor * iqr)
            
        logger.info(f"Calculated outlier bounds for {len(self.lower_bounds_)} features.")
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, lower in self.lower_bounds_.items():
            upper = self.upper_bounds_[col]
            # Clip values
            X_copy[col] = np.clip(X_copy[col], a_min=lower, a_max=upper)
        
        logger.info("Outliers handled via IQR clipping.")
        return X_copy

# ---------------------------------------------------------
# Execution (Can be run standalone to generate interim data)
# ---------------------------------------------------------
def run_preprocessing():
    input_path = Path(r"data/raw_data/airline_financial_impact.csv")
    output_dir = Path(r"data/interim_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Initialize and run Transformers manually for interim saving
    cleaner = DataCleaner()
    df_cleaned = cleaner.fit_transform(df)
    
    # We will exclude target variables from outlier handling to avoid data leakage
    features_to_clip = [c for c in df_cleaned.select_dtypes(include=np.number).columns if c not in ['net_profit_usd_m', 'profit_margin_pct']]
    
    outlier_handler = OutlierHandler(columns=features_to_clip)
    df_processed = outlier_handler.fit_transform(df_cleaned)
    
    output_path = output_dir / "preprocessed_data.csv"
    df_processed.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing()