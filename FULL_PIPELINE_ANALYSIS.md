# Complete Pipeline Analysis: EDA → Preprocessing → Feature Engineering

## Overview
This document traces the complete data science workflow from exploratory data analysis through preprocessing and feature engineering to the final modeling pipeline.

---

## 1. EDA Analysis (`eda/eda_analysis.py`)

### Data Loading
- **Source**: `data/raw_data/airline_financial_impact.csv`
- **Initial columns dropped**: `quarter`, `month` (likely redundant with date column)

### Key EDA Findings

#### 1.1 Normality Testing (KS Test Results)
**ALL 13 numeric features are NOT normally distributed** (p < 0.05):

| Feature | D_stat | p_value | Distribution |
|---------|--------|---------|--------------|
| fleet_size | 0.2384 | 0.0 | Not Normal |
| revenue_usd_m | 0.1931 | 0.0 | Not Normal |
| fuel_cost_usd_m | 0.1121 | 0.0 | Not Normal |
| fuel_cost_pct_revenue | 0.164 | 0.0 | Not Normal |
| net_profit_usd_m | 0.1598 | 0.0 | Not Normal |
| profit_margin_pct | 0.3129 | 0.0 | Not Normal |
| passengers_carried_m | 0.2194 | 0.0 | Not Normal |
| fuel_hedging_pct | 0.0599 | 0.0105 | Not Normal |
| hedge_savings_usd_m | 0.3456 | 0.0 | Not Normal |
| brent_crude_usd_barrel | 0.1249 | 0.0 | Not Normal |
| jet_fuel_usd_barrel | 0.1251 | 0.0 | Not Normal |
| daily_fuel_consumption_bbl | 0.1088 | 0.0 | Not Normal |
| quarterly_fuel_bbl | 0.1088 | 0.0 | Not Normal |

**Impact**: Non-normal distributions suggest:
- Need for robust scaling methods
- Potential benefit from log/power transformations
- Outlier handling is critical

#### 1.2 Correlation Analysis
- Generated correlation heatmap for all numeric features
- Identified target correlation with `net_profit_usd_m`
- Detected high multicollinearity (>0.85) between feature pairs

**Key Insights**:
- `daily_fuel_consumption_bbl` and `quarterly_fuel_bbl` are likely highly correlated (0.1088 D_stat identical)
- Fuel costs strongly related to revenue metrics

#### 1.3 Skewness Analysis
- All features analyzed for skewness
- Visualized with threshold lines at ±1 (high skew indicators)

#### 1.4 Outlier Detection
- Boxplots generated for all 13 numeric features
- Identified outliers in multiple features (visible in boxplots)

#### 1.5 Categorical Analysis
Categorical features analyzed:
- `airline`
- `airline_type`
- `conflict_phase`
- `country`
- `region`

**Bivariate Analysis**:
- KDE plots: numeric features by categorical groups
- Cross-tabulations: categorical vs categorical relationships
- Filtered high-cardinality features (>10 unique values)

---

## 2. Preprocessing Pipeline (`preprocessing/preprocessing.py`)

### 2.1 Custom Transformers (sklearn-compatible)

#### DropColumns
```python
DropColumns(cols=["unnecessary_column"])
```
- Removes specified columns
- Currently configured for generic "unnecessary_column"
- **Connection to EDA**: Should drop `quarter`, `month` as done in EDA

#### FillMissing
```python
FillMissing(strategy="median")
```
- Handles missing values in numeric columns
- Uses median (robust to outliers)
- **Connection to EDA**: Median chosen because distributions are NOT normal

#### DateFeatureExtractor
```python
DateFeatureExtractor(date_column="date")
```
- Extracts: `year`, `month`
- **Cyclical encoding**: `month_sin`, `month_cos`
- **Connection to EDA**: Replaces dropped `quarter`, `month` with engineered features

#### OutlierClipper
```python
OutlierClipper()
```
- IQR method: clips at Q1 - 1.5×IQR and Q3 + 1.5×IQR
- **Connection to EDA**: Directly addresses outliers identified in boxplots

#### FeatureCreator
```python
FeatureCreator()
```
- Creates ratio features:
  - `price_per_km = airline_price / distance`
  - `oil_price_impact = oil_price / airline_price`
- **Connection to EDA**: Leverages correlation insights

#### Scaler
```python
Scaler()  # StandardScaler wrapper
```
- Standardizes all numeric features
- **Connection to EDA**: Necessary due to non-normal distributions and varying scales

### 2.2 Pipeline Execution
```python
Pipeline([
    ("drop", DropColumns(...)),
    ("missing", FillMissing(...)),
    ("date", DateFeatureExtractor(...)),
    ("outliers", OutlierClipper()),
    ("features", FeatureCreator()),
    ("scaling", Scaler())
])
```

---

## 3. Feature Engineering (`feature_engineering/feature_engineering.py`)

### 3.1 Advanced Transformers

#### LogTransformer
```python
LogTransformer(columns=["revenue_usd_m", "oil_price"])
```
- Creates `{col}_log` features
- **Connection to EDA**: Addresses skewness and non-normality
- **Specific to**: revenue_usd_m (D_stat=0.1931), oil prices

#### RatioFeatures
```python
RatioFeatures()
```
- `profit_margin = net_profit_usd_m / revenue_usd_m`
- `cost_efficiency = operating_cost / revenue_usd_m`
- **Connection to EDA**: Leverages correlation between profit and revenue metrics

#### InteractionFeatures
```python
InteractionFeatures(pairs=[("oil_price", "revenue_usd_m")])
```
- Creates multiplicative interactions
- **Connection to EDA**: Captures non-linear relationships identified in scatter plots

#### TimeFeatures
```python
TimeFeatures(date_col="date")
```
- Extracts: `year`, `month`, `quarter`
- Cyclical encoding: `month_sin`, `month_cos`
- **Connection to EDA**: Enhanced version of DateFeatureExtractor

#### RollingFeatures
```python
RollingFeatures(column="oil_price", window=3)
```
- `{col}_rolling_mean`, `{col}_rolling_std`
- **Connection to EDA**: Captures temporal trends in fuel prices

#### FrequencyEncoder
```python
FrequencyEncoder(columns=["airline"])
```
- Encodes categorical by frequency
- **Connection to EDA**: Handles high-cardinality `airline` feature (>10 unique values)

#### BinningFeatures
```python
BinningFeatures(column="oil_price", bins=5)
```
- Creates discrete bins
- **Connection to EDA**: Captures non-linear effects of oil price on profits

---

## 4. Complete Pipeline Integration (`pipeline/run_pipeline.py`)

### 4.1 Sequential Execution
```python
def run_full_pipeline():
    # 1. EDA → Load Data
    df = load_data()
    
    # 2. Preprocessing
    df = process_data(df)
    
    # 3. Feature Engineering
    df = apply_feature_engineering(df)
    
    return df
```

### 4.2 Training Pipeline (`pipeline/train_pipeline.py`)

#### Column Definitions
```python
num_cols = ["oil_price", "distance"]
cat_cols = ["airline", "source"]
```

#### ColumnTransformer
```python
ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
```

#### Full Training Pipeline
```python
Pipeline([
    ("drop_cols", DropColumns(cols=["id"])),
    ("date_features", DateFeatureExtractor(date_column="date")),
    ("missing", FillMissing(strategy="median", columns=num_cols)),
    ("outliers", OutlierClipper(columns=num_cols)),
    ("feature_creation", FeatureCreator()),
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])
```

---

## 5. Critical Connections: EDA → Implementation

### 5.1 Non-Normal Distributions
**EDA Finding**: All features non-normal (p < 0.05)

**Implementation**:
- ✅ `FillMissing(strategy="median")` - robust to skewness
- ✅ `LogTransformer` - normalizes skewed distributions
- ✅ `OutlierClipper` - handles extreme values
- ✅ `StandardScaler` - standardizes after transformations

### 5.2 High Skewness
**EDA Finding**: Multiple features with |skew| > 1

**Implementation**:
- ✅ `LogTransformer(columns=["revenue_usd_m", "oil_price"])`
- ✅ `BinningFeatures` - discretizes skewed continuous variables

### 5.3 Outliers
**EDA Finding**: Boxplots show outliers in all numeric features

**Implementation**:
- ✅ `OutlierClipper()` - IQR-based clipping
- ✅ Applied BEFORE scaling to prevent outlier influence

### 5.4 Multicollinearity
**EDA Finding**: High correlation (>0.85) between feature pairs

**Implementation**:
- ⚠️ `DropColumns` - should drop one of correlated pairs
- ⚠️ Consider VIF analysis or PCA

### 5.5 Temporal Patterns
**EDA Finding**: Time-based features (`quarter`, `month`)

**Implementation**:
- ✅ `DateFeatureExtractor` - cyclical encoding
- ✅ `TimeFeatures` - enhanced temporal features
- ✅ `RollingFeatures` - captures trends

### 5.6 Categorical Handling
**EDA Finding**: 5 categorical features, some high-cardinality

**Implementation**:
- ✅ `FrequencyEncoder` - for high-cardinality `airline`
- ✅ `OneHotEncoder` - for low-cardinality features
- ✅ Filtered features with >10 unique values in EDA

### 5.7 Feature Interactions
**EDA Finding**: Scatter plots show non-linear relationships

**Implementation**:
- ✅ `InteractionFeatures` - multiplicative interactions
- ✅ `RatioFeatures` - domain-specific ratios
- ✅ `BinningFeatures` - captures non-linearity

---

## 6. Recommendations & Gaps

### 6.1 Missing Connections

#### Issue 1: Column Name Mismatch
**Problem**: 
- EDA uses actual column names from dataset
- Preprocessing/FE use generic names (`oil_price`, `airline_price`, `distance`)

**Solution**: Update preprocessing to use actual column names:
```python
# Should be:
num_cols = ["brent_crude_usd_barrel", "jet_fuel_usd_barrel", "revenue_usd_m", ...]
```

#### Issue 2: Multicollinearity Not Addressed
**EDA Finding**: High correlation pairs (>0.85)

**Missing**: 
- No VIF calculation
- No automatic feature selection
- No PCA option

**Solution**: Add VIF-based feature selection:
```python
class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10):
        self.threshold = threshold
    # ... implementation
```

#### Issue 3: Target Leakage Risk
**Problem**: `profit_margin_pct` and `net_profit_usd_m` are in features

**Solution**: Ensure target-related features are dropped:
```python
DropColumns(cols=["net_profit_usd_m", "profit_margin_pct"])
```

#### Issue 4: Incomplete Feature Engineering
**EDA Shows**: 13 numeric features

**FE Only Transforms**: 2 features (`revenue_usd_m`, `oil_price`)

**Solution**: Apply transformations to all skewed features:
```python
LogTransformer(columns=[
    "fleet_size", "revenue_usd_m", "fuel_cost_usd_m",
    "passengers_carried_m", "hedge_savings_usd_m"
])
```

### 6.2 Data Leakage Concerns

#### Temporal Leakage
**Issue**: `RollingFeatures` uses future data if not sorted by date

**Solution**:
```python
# Add before rolling features
df = df.sort_values('date')
```

#### Train-Test Contamination
**Issue**: Scaling fitted on entire dataset

**Solution**: Use sklearn Pipeline with proper train/test split

### 6.3 Performance Optimizations

#### Redundant Transformations
**Issue**: Both `DateFeatureExtractor` and `TimeFeatures` do similar work

**Solution**: Consolidate into single transformer

#### Unnecessary Features
**Issue**: `daily_fuel_consumption_bbl` and `quarterly_fuel_bbl` have identical D_stat

**Solution**: Drop one (likely perfectly correlated)

---

## 7. Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA                                  │
│         data/raw_data/airline_financial_impact.csv          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  EDA ANALYSIS                                │
│              (eda/eda_analysis.py)                          │
├─────────────────────────────────────────────────────────────┤
│ • Drop quarter, month                                        │
│ • Normality tests → ALL non-normal                          │
│ • Correlation analysis → High multicollinearity             │
│ • Skewness analysis → Multiple features |skew| > 1          │
│ • Outlier detection → Present in all features               │
│ • Categorical analysis → 5 cat features                     │
│ • Bivariate analysis → Non-linear relationships             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 PREPROCESSING                                │
│           (preprocessing/preprocessing.py)                   │
├─────────────────────────────────────────────────────────────┤
│ 1. DropColumns → Remove unnecessary features                │
│ 2. FillMissing(median) → Handle missing (robust)            │
│ 3. DateFeatureExtractor → Cyclical encoding                 │
│ 4. OutlierClipper → IQR-based clipping                      │
│ 5. FeatureCreator → Basic ratio features                    │
│ 6. Scaler → StandardScaler                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                             │
│        (feature_engineering/feature_engineering.py)          │
├─────────────────────────────────────────────────────────────┤
│ 1. LogTransformer → Address skewness                        │
│ 2. RatioFeatures → Domain-specific ratios                   │
│ 3. InteractionFeatures → Capture non-linearity              │
│ 4. TimeFeatures → Enhanced temporal features                │
│ 5. RollingFeatures → Trend capture                          │
│ 6. FrequencyEncoder → High-cardinality handling             │
│ 7. BinningFeatures → Discretization                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                TRAINING PIPELINE                             │
│            (pipeline/train_pipeline.py)                      │
├─────────────────────────────────────────────────────────────┤
│ • ColumnTransformer (num + cat)                             │
│ • Model training (LinearRegression)                         │
│ • Model persistence (joblib)                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  SAVED MODEL                                 │
│              models/model.pkl                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Summary

### Strengths
✅ Comprehensive EDA with statistical rigor (KS tests, correlation, skewness)
✅ Modular sklearn-compatible transformers
✅ Proper handling of non-normal distributions
✅ Cyclical encoding for temporal features
✅ Robust outlier handling
✅ Feature engineering addresses EDA findings

### Areas for Improvement
⚠️ Column name alignment between EDA and preprocessing
⚠️ Multicollinearity not addressed in pipeline
⚠️ Potential target leakage
⚠️ Incomplete transformation coverage
⚠️ Temporal leakage risk in rolling features
⚠️ Redundant transformers (DateFeatureExtractor vs TimeFeatures)

### Next Steps
1. Align column names across all modules
2. Add VIF-based feature selection
3. Implement proper train/test splitting
4. Extend transformations to all skewed features
5. Add data validation checks
6. Create comprehensive unit tests
7. Add model evaluation metrics
8. Implement cross-validation
