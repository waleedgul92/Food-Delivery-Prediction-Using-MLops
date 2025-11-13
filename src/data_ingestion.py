import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import yaml
import mlflow
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from mlflow_config import setup_mlflow_experiment, get_parent_experiment_name

load_dotenv()


def setup_logging():
    """Sets up a logger for the script."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        os.makedirs('log_data', exist_ok=True)
        logger.addHandler(logging.FileHandler('log_data/data_ingestion.log'))
        logger.addHandler(logging.StreamHandler())
        
    return logger


logger = setup_logging()


def fetch_csv_from_s3(bucket_name, file_key, region_name='us-east-1'):
    """Fetch a CSV file from S3 and return as a pandas DataFrame."""
    try:
        logger.info(f"Connecting to S3 bucket: {bucket_name} in region: {region_name}")
        s3_client = boto3.client('s3', region_name=region_name)
        logger.info(f"Fetching file: s3://{bucket_name}/{file_key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_csv(response['Body'])
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from S3")
        return df
    
    except NoCredentialsError:
        logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file.")
        raise
    except ClientError as e:
        logger.error(f"AWS ClientError: {e}")
        raise
    except Exception as e:
        logger.error(f"Error fetching file from S3: {e}")
        raise


def load_params():
    """Loads parameters from the params.yaml file."""
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)


def split_and_save_data(params, logger):
    """Loads data from S3, saves raw data, splits it, and saves train/test data locally."""
    
    logger.info("Starting the data ingestion process from S3.")
    
    try:
        bucket_name = params['data']['s3_bucket']
        file_key = params['data']['s3_key']
        region_name = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    except KeyError as e:
        logger.error(f"Missing key in params.yaml: {e}")
        raise
    
    raw_dir = params['data']['raw_dir']
    interim_dir = params['data']['interim_dir']
    test_size = params['data_split']['test_size']
    random_state = params['base']['random_state']
    
    s3_path = f"s3://{bucket_name}/{file_key}"
    logger.info(f"Attempting to read data from: {s3_path}")
    
    try:
        df = fetch_csv_from_s3(bucket_name, file_key, region_name=region_name)
        logger.info(f"Successfully loaded {df.shape[0]} rows from {s3_path}.")
        mlflow.log_param("s3_source_path", s3_path) 
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])
    except Exception as e:
        logger.error(f"Failed to read file from {s3_path}. Error: {e}")
        raise

    # Save raw data to data/raw directory
    os.makedirs(raw_dir, exist_ok=True)
    raw_output_path = os.path.join(raw_dir, 'dataset.csv')
    df.to_csv(raw_output_path, index=False)
    logger.info(f"Raw data saved to {raw_output_path}")
    mlflow.log_artifact(raw_output_path, "raw_data")

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    logger.info(f"Data split complete. Training set size: {len(train_data)}, Test set size: {len(test_data)}")

    # Save split datasets to data/interim directory
    os.makedirs(interim_dir, exist_ok=True)
    
    train_output_path = os.path.join(interim_dir, params['data']['train_csv'])
    test_output_path = os.path.join(interim_dir, params['data']['test_csv'])

    train_data.to_csv(train_output_path, index=False)
    test_data.to_csv(test_output_path, index=False)
    logger.info(f"Train data saved to {train_output_path}")
    logger.info(f"Test data saved to {test_output_path}")
    
    mlflow.log_artifact(train_output_path, "split_data")
    mlflow.log_artifact(test_output_path, "split_data")
    
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    logger.info("Data ingestion process complete.")


if __name__ == "__main__":
    try:
        params = load_params()
        
        # Set up MLflow under parent experiment
        stage = setup_mlflow_experiment("data_ingestion")
        
        # Start run with stage-specific name and tags
        with mlflow.start_run(run_name="data_ingestion"):
            mlflow.set_tag("stage", stage)
            mlflow.set_tag("pipeline_step", "1_data_ingestion")
            split_and_save_data(params, logger)
            
    except Exception as e:
        logger.critical(f"An error occurred in the main execution block: {e}")