import pandas as pd
from pathlib import Path

def view_best_models():
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent 
    registry_path = PROJECT_ROOT / "models" / "model_csv" /"model_registry.csv"
    
    if not registry_path.exists():
        print("❌ No registry found. Train a model first!")
        return
        
    df = pd.read_csv(registry_path)
    
    print("\n" + "="*60)
    print("🏆 MODEL REGISTRY LEADERBOARD 🏆")
    print("="*60)
    
    # Sort models by lowest RMSE (Best performing first)
    best_models = df.sort_values(by='rmse', ascending=True)[['version', 'status', 'rmse', 'r2_score', 'date_trained']]
    
    print(best_models.to_string(index=False))
    print("="*60)
    print(f"Total Models Tracked: {len(df)}")
    
    # Find the absolute best model
    best_version = best_models.iloc[0]['version']
    print(f"\n🚀 Recommendation: Promote {best_version} to 'Production' based on lowest RMSE.\n")

if __name__ == "__main__":
    view_best_models()