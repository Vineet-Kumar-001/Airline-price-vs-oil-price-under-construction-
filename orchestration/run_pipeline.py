import subprocess
import logging
import sys
import time
import os
from pathlib import Path

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - PipelineOrchestrator - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Pipeline Configuration & Path Mapping
# ---------------------------------------------------------
# 1. Find the orchestration folder
CURRENT_DIR = Path(__file__).resolve().parent

# 2. Go up one level to find the main project root ('Airline_price vs oil price')
PROJECT_ROOT = CURRENT_DIR.parent 

# 3. Define the exact path for each script based on the new folders
PIPELINE_SCRIPTS = [
    PROJECT_ROOT / "eda" / "eda_analysis.py",
    PROJECT_ROOT / "preprocessing" / "preprocessing.py",
    PROJECT_ROOT / "feature_engineering" / "feature_engineering.py",
    PROJECT_ROOT / "models" / "model_train.py",
    PROJECT_ROOT / "optimization" / "hyperparameter_tuning.py",
    PROJECT_ROOT / "models" / "model_registry" / "view_registry.py"

]

def run_script(script_path: Path) -> bool:
    """
    Executes a single Python script using subprocess.
    """
    if not script_path.exists():
        logger.error(f"❌ Script not found: {script_path}")
        logger.error("Please check your folder names and ensure the file is inside.")
        return False

    script_name = script_path.name
    logger.info(f"{'='*50}")
    logger.info(f"🚀 STARTING STEP: {script_name}")
    logger.info(f"📁 Location: {script_path.parent.name}/{script_name}")
    logger.info(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # sys.executable ensures we use your virtual environment (.venv)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            text=True,
            capture_output=False 
        )
        
        execution_time = time.time() - start_time
        logger.info(f"✅ COMPLETED STEP: {script_name} (Took {execution_time:.2f} seconds)\n")
        return True
        
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ FAILED STEP: {script_name} (Failed after {execution_time:.2f} seconds)")
        logger.error(f"Return code: {e.returncode}")
        return False

def main():
    """
    Main orchestrator function that runs the pipeline sequentially.
    """
    logger.info("Starting Modular End-to-End Machine Learning Pipeline...")
    total_start_time = time.time()
    
    for script_path in PIPELINE_SCRIPTS:
        success = run_script(script_path)
        
        if not success:
            logger.critical("🚨 Pipeline execution HALTED due to an error.")
            sys.exit(1)
            
    total_time = time.time() - total_start_time
    logger.info(f"{'='*50}")
    logger.info(f"🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"⏱️ Total Execution Time: {total_time:.2f} seconds")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    # CRITICAL: Change the working directory to the PROJECT ROOT before running.
    # This ensures that when the scripts look for "data\raw_data\...", they find it!
    os.chdir(PROJECT_ROOT) 
    main()