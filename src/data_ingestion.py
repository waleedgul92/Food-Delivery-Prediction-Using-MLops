# src/data_ingestion.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import yaml
import mlflow
import dagshub
import s3fs  # <-- ADDED: For reading from S3

# --- Dagshub/MLflow setup (UNCHANGED) ---
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "hwaleed0035"
repo_name = "Food-Delivery-Prediction-Using-MLops"
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


def setup_logging():
    # --- (UNCHANGED) ---
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.FileHandler('log_data/data_split.log'))
        logger.addHandler(logging.StreamHandler())
    return logger

def load_params():
    # --- (UNCHANGED) ---
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)

def split_and_save_data(params, logger):
    """Loads data from S3, splits, and saves it locally."""
    with mlflow.start_run(run_name="data_ingestion"):
        logger.info("Starting the data ingestion process from S3.")
        
        # --- MODIFIED: Read S3 path from params ---
        try:
            bucket_name = params['data']['s3_bucket']
            file_key = params['data']['s3_key']
            s3_path = f"s3://{bucket_name}/{file_key}"
        except KeyError:
            logger.error("Failed to find 's3_bucket' or 's3_key' in params.yaml")
            raise
        
        output_dir = params['data']['interim_dir']
        test_size = params['data_split']['test_size']
        random_state = params['base']['random_state']
        
        logger.info(f"Attempting to read data from: {s3_path}")
        
        try:
            # This relies on AWS credentials (AWS_ACCESS_KEY_ID, etc.)
            # being set as environment variables, which your CI workflow does.
            df = pd.read_csv(s3_path)
            logger.info(f"Successfully loaded {file_key} from S3.")
            
            # --- MODIFIED: Log the S3 path as a parameter, not an artifact ---
            mlflow.log_param("s3_source_path", s3_path) 
        except Exception as e:
            # Catching a broader exception since S3 errors can vary
            # (e.g., FileNotFoundError, NoCredentialsError)
            logger.error(f"Failed to read file from {s3_path}. Error: {e}")
            raise

        # --- Splitting and saving logic (UNCHANGED) ---
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        os.makedirs(output_dir, exist_ok=True)
        
        train_output_path = os.path.join(output_dir, params['data']['train_csv'])
        test_output_path = os.path.join(output_dir, params['data']['test_csv'])

        train_data.to_csv(train_output_path, index=False)
        test_data.to_csv(test_output_path, index=False)
        
        # --- MLflow logging (UNCHANGED) ---
        mlflow.log_artifact(train_output_path, "split_data")
        mlflow.log_artifact(test_output_path, "split_data")
        
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        logger.info("Process complete.")

if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    split_and_save_data(params, logger)