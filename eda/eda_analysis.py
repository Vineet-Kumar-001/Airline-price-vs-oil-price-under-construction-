import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
from scipy import stats
import warnings
warnings.filterwarnings("ignore")



# 1. Load the dataset
def load_data():
    df = pd.read_csv(r"data\raw_data\synthetic_airline_data_100.csv")
    return df
df = load_data()


# 2. Basic Dataset Overview
#print("--- First 5 rows of the dataset ---")

def dataset_overview(df):
    print("Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())
    print("\nData Types:\n", df.dtypes)

df = df.drop('quarter',axis=1)
df = df.drop('month',axis=1)

cat_cols_eda =df.select_dtypes(include=['object'])
num_cols_eda = df.select_dtypes(include=['int64', 'float64'])

save_path = r"eda\graphs\Univariate_analysis"
os.makedirs(save_path, exist_ok=True)

print("Saving plots in:", save_path)

plt.style.use("dark_background")

for col in num_cols_eda.columns:
    plt.figure(figsize=(6,4))
    
    plt.hist(df[col], bins=20, edgecolor='orange')
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    file_name = os.path.join(save_path, f"{col}.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.close()

print("✅ All graphs of univarite analysis saved successfully!")


save_path = r"eda\graphs\Boxplot"
os.makedirs(save_path, exist_ok=True)

plt.style.use("dark_background")

for col in num_cols_eda.columns:
    plt.figure(figsize=(5,4))
    
    plt.boxplot(df[col])
    plt.title(col)
    plt.ylabel("Value")
    
    # Save file
    file_name = os.path.join(save_path, f"{col}_boxplot.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.close()

print("✅ All boxplots saved successfully!")



plt.style.use("dark_background")

# All column pairs
pairs = list(itertools.combinations(num_cols_eda.columns, 2))

# Save path
save_path = r"eda\graphs\Scatterplots"
os.makedirs(save_path, exist_ok=True)

for c1, c2 in pairs:
    plt.figure(figsize=(5,4))
    
    plt.scatter(df[c1], df[c2], alpha=0.6, s=10)
    plt.title(f"{c1} vs {c2}")
    plt.xlabel(c1)
    plt.ylabel(c2)
    
    # Safe filename (remove spaces)
    safe_c1 = c1.replace(" ", "_")
    safe_c2 = c2.replace(" ", "_")
    
    file_name = os.path.join(save_path, f"{safe_c1}_vs_{safe_c2}.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.close()

print("✅ All scatter plots saved!")




save_path = r"eda\graphs\Correlation"
os.makedirs(save_path, exist_ok=True)

plt.style.use("dark_background")

plt.figure(figsize=(14, 10)) 

sns.heatmap(
    num_cols_eda.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    annot_kws={"size": 8}
)

plt.xticks(rotation=90, ha='right')  
plt.yticks(rotation=0)

plt.tight_layout()

# Save BEFORE show 🔥
file_name = os.path.join(save_path, "correlation_heatmap.png")
plt.savefig(file_name, dpi=300, bbox_inches='tight')

plt.show()
plt.close()

print("✅ Heatmap saved successfully!")



# Compute correlation
corr = num_cols_eda.corr()
target_corr = corr[['net_profit_usd_m']].sort_values(
    by='net_profit_usd_m', ascending=False
)

# Save path
save_path = r"eda\graphs\Correlation"
os.makedirs(save_path, exist_ok=True)

plt.style.use("dark_background")

plt.figure(figsize=(6, 10))

sns.heatmap(
    target_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    annot_kws={"size": 9}
)

plt.title("Correlation with Net Profit", fontsize=12)
plt.yticks(rotation=0)

plt.tight_layout()

# Save BEFORE show 🔥
file_name = os.path.join(save_path, "target_correlation_net_profit.png")
plt.savefig(file_name, dpi=300, bbox_inches='tight')

plt.show()
plt.close()

print("✅ Target correlation heatmap saved!")


corr = num_cols_eda.corr()

high_corr_pairs = (
    corr.abs()
    .unstack()
    .sort_values(ascending=False)
)

high_corr_pairs = high_corr_pairs[
    (high_corr_pairs > 0.85) & (high_corr_pairs < 1)
]

print(high_corr_pairs)


# Compute skewness
skewness = df.select_dtypes(include='number').skew()

# Save path
save_path = r"eda\graphs\Skewness"
os.makedirs(save_path, exist_ok=True)

plt.style.use("dark_background")

plt.figure(figsize=(10,6))

sns.barplot(x=skewness.values, y=skewness.index)

plt.axvline(0, color='white', linestyle='--')
plt.axvline(1, color='red', linestyle='--')   # high skew
plt.axvline(-1, color='red', linestyle='--')

plt.title("Feature Skewness")
plt.xlabel("Skewness Value")
plt.ylabel("Features")

plt.tight_layout()

# Save
plt.savefig(os.path.join(save_path, "skewness_barplot.png"), dpi=300)
plt.show()
plt.close()

# Select numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Save path
save_path = r"eda\graphs\Normality"
os.makedirs(save_path, exist_ok=True)

plt.style.use("dark_background")

for col in num_cols:
    data = df[col].dropna()

    # Skip constant columns
    if data.nunique() <= 1:
        continue

    # Standardize for KS test
    standardized = (data - data.mean()) / data.std()

    # KS Test
    ks_stat, p_value = stats.kstest(standardized, 'norm')

    # -------- Histogram + Normal Curve --------
    plt.figure(figsize=(6,4))

    sns.histplot(data, kde=True, stat='density')

    x = np.linspace(data.min(), data.max(), 100)
    plt.plot(x, stats.norm.pdf(x, data.mean(), data.std()), color='red')

    plt.title(f"{col} | KS p={p_value:.4f}")

    file_name = os.path.join(save_path, f"{col}_ks.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

    # -------- Q-Q Plot --------
    plt.figure(figsize=(5,5))

    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot: {col}")

    file_name = os.path.join(save_path, f"{col}_qq.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

print("✅ KS test + Normality plots saved for all numeric columns!")


save_path = r"eda\graphs\Normality"
os.makedirs(save_path, exist_ok=True)

print("Saving files in:", save_path)

plt.style.use("dark_background")

# -------------------------------
# 3. Store results
# -------------------------------
results = []

# -------------------------------
# 4. Loop through columns
# -------------------------------
for col in num_cols_eda:
    data = df[col].dropna()

    # Skip useless columns
    if data.nunique() <= 1:
        continue

    # Standardize data (IMPORTANT for KS test)
    standardized = (data - data.mean()) / data.std()

    # KS Test
    D_stat, p_value = stats.kstest(standardized, 'norm')

    result = "Normal" if p_value > 0.05 else "Not Normal"

    # Save result
    results.append({
        "Feature": col,
        "D_stat": round(D_stat, 4),
        "p_value": round(p_value, 4),
        "Result": result
    })

    # -------------------------------
    # 5. Plot
    # -------------------------------
    plt.figure(figsize=(6,4))

    sns.histplot(data, kde=True, stat='density')

    x = np.linspace(data.min(), data.max(), 100)
    plt.plot(x, stats.norm.pdf(x, data.mean(), data.std()), color='red')

    plt.title(f"{col} | p={p_value:.4f}")
    plt.xlabel(col)
    plt.ylabel("Density")

    # Save plot
    file_name = os.path.join(save_path, f"{col}_ks.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')

    plt.close()

# -------------------------------
# 6. Save summary CSV
# -------------------------------
results_df = pd.DataFrame(results)

# -------------------------------
# Create summary_csv folder inside data
# -------------------------------
summary_path = r"data\summary_kstest_eda_csv"

os.makedirs(summary_path, exist_ok=True)

# -------------------------------
# Save CSV
# -------------------------------
csv_path = os.path.join(summary_path, "ks_test_summary.csv")

results_df.to_csv(csv_path, index=False)

print("✅ CSV saved at:", csv_path)
results_df.to_csv(csv_path, index=False)

# -------------------------------
# 7. Print results
# -------------------------------
print("✅ All plots + CSV saved successfully!")
print(results_df)

# -------------------------------
# 8. Optional: Non-normal features
# -------------------------------
non_normal = results_df[results_df["Result"] == "Not Normal"]["Feature"]
print("❌ Non-normal columns:", list(non_normal))



# Save path
save_path = r"eda\graphs\Categorical"
os.makedirs(save_path, exist_ok=True)

plt.style.use("dark_background")

for col in cat_cols_eda.columns:
    plt.figure(figsize=(6,4))
    
    # Countplot (correct for categorical)
    sns.countplot(x=df[col], order=df[col].value_counts().index)
    
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Count")
    
    plt.xticks(rotation=90)
    
    # Safe filename
    safe_col = col.replace(" ", "_")
    file_name = os.path.join(save_path, f"{safe_col}_countplot.png")
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.close()

print("✅ All categorical plots saved!")


# Save path
save_path = r"eda\graphs\Bivariate_KDE"
os.makedirs(save_path, exist_ok=True)

plt.style.use("dark_background")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for num_col in num_cols:
    for cat_col in cat_cols:
        
        # Skip high-cardinality categorical columns
        if df[cat_col].nunique() > 10:
            continue
        
        plt.figure(figsize=(6,4))
        
        sns.kdeplot(
            data=df,
            x=num_col,
            hue=cat_col,
            fill=True,
            common_norm=False,
            alpha=0.4
        )
        
        plt.title(f"{num_col} by {cat_col}")
        
        # Safe filename
        safe_num = num_col.replace(" ", "_")
        safe_cat = cat_col.replace(" ", "_")
        
        file_name = os.path.join(save_path, f"{safe_num}_by_{safe_cat}.png")
        
        plt.tight_layout()
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()

print("✅ All KDE bivariate plots saved!")



save_path = r"eda\graphs\Categorical_vs_Categorical"
os.makedirs(save_path, exist_ok=True)

plt.style.use("dark_background")

for col1 in cat_cols_eda.columns:
    for col2 in cat_cols_eda.columns:
        
        if col1 == col2:
            continue
        
        # Skip high-cardinality columns
        if df[col1].nunique() > 10 or df[col2].nunique() > 10:
            continue
        
        ct = pd.crosstab(df[col1], df[col2])
        
        plt.figure(figsize=(6,4))
        
        sns.heatmap(ct, annot=True, fmt='d', cmap='coolwarm')
        
        plt.title(f"{col1} vs {col2}")
        
        # Safe filename
        safe1 = col1.replace(" ", "_")
        safe2 = col2.replace(" ", "_")
        
        file_name = os.path.join(save_path, f"{safe1}_vs_{safe2}.png")
        
        plt.tight_layout()
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()

print("✅ All categorical vs categorical heatmaps saved!")

