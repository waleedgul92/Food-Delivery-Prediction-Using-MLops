# src/data_ingestion.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import yaml
import mlflow
import dagshub
dagshub.init(repo_owner='hwaleed0035', repo_name='Food-Delivery-Prediction-Using-MLops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/hwaleed0035/Food-Delivery-Prediction-Using-MLops.mlflow")

mlflow.set_experiment("Food Delivery Prediction Model Training")

def setup_logging():
    """Configures logging."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.FileHandler('log_data/data_split.log'))
        logger.addHandler(logging.StreamHandler())
    return logger

def load_params():
    """Loads parameters from params.yaml"""
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)

def split_and_save_data(params, logger):
    """Loads, splits, and saves the dataset based on params."""
    with mlflow.start_run(run_name="data_ingestion"):
        logger.info("Starting the data splitting process.")
        
        input_path = params['data']['raw_dataset']
        output_dir = params['data']['interim_dir']
        test_size = params['data_split']['test_size']
        random_state = params['base']['random_state']
        
        try:
            df = pd.read_csv(input_path)
            mlflow.log_artifact(input_path, "raw_data")
        except FileNotFoundError:
            logger.error(f"File not found at {input_path}")
            return

        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        os.makedirs(output_dir, exist_ok=True)
        
        train_output_path = os.path.join(output_dir, params['data']['train_csv'])
        test_output_path = os.path.join(output_dir, params['data']['test_csv'])

        train_data.to_csv(train_output_path, index=False)
        test_data.to_csv(test_output_path, index=False)
        
        mlflow.log_artifact(train_output_path, "split_data")
        mlflow.log_artifact(test_output_path, "split_data")
        
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        logger.info("Process complete.")

if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    split_and_save_data(params, logger)