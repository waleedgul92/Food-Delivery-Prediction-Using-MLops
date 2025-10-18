# src/mlflow_config.py
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

# Load and validate DagsHub token
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# Set up MLflow tracking credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# DagsHub configuration
dagshub_url = "https://dagshub.com"
repo_owner = "hwaleed0035"
repo_name = "Food-Delivery-Prediction-Using-MLops"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Parent experiment name
PARENT_EXPERIMENT = "Food Delivery Prediction Model Training"

# Pipeline stages
STAGES = {
    "data_ingestion": "Data Ingestion",
    "data_processing": "Data Processing",
    "feature_engineering": "Feature Engineering",
    "train_model": "Model Training",
    "test_model": "Model Evaluation"
}


def setup_mlflow_experiment(stage_name):
    """
    Sets up MLflow for a specific pipeline stage.
    All runs are logged under a single parent experiment.
    Stages are distinguished using tags and run names.
    
    Args:
        stage_name (str): One of 'data_ingestion', 'data_processing', 'feature_engineering', 
                         'train_model', 'test_model'
    
    Returns:
        str: The stage name (used for tagging)
    
    Example:
        >>> setup_mlflow_experiment("data_ingestion")
        'data_ingestion'
    """
    if stage_name not in STAGES:
        raise ValueError(f"Unknown stage: {stage_name}. Must be one of {list(STAGES.keys())}")
    
    # Set parent experiment for all runs
    mlflow.set_experiment(PARENT_EXPERIMENT)
    
    return stage_name


def get_parent_experiment_name():
    """Returns the parent experiment name."""
    return PARENT_EXPERIMENT