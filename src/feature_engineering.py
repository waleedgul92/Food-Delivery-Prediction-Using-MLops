# src/feature_engineering.py
import os
import pandas as pd
import numpy as np
import joblib
import json
import boto3
from botocore.exceptions import ClientError
from sklearn.preprocessing import MinMaxScaler
import logging
import yaml
import mlflow
from dotenv import load_dotenv
from mlflow_config import setup_mlflow_experiment

load_dotenv()


def setup_logging():
    """Configures logging."""
    os.makedirs('log_data', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('log_data/feature_engineering.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_params():
    """Loads parameters from params.yaml"""
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)


def normalize_data(train_df, test_df, target_col, logger, params):
    """Fits a scaler on training data, logs it, and scales both sets."""
    logger.info("Starting numerical data normalization.")
    scaler = MinMaxScaler()
    train_normalized, test_normalized = train_df.copy(), test_df.copy()

    numerical_cols = train_df.select_dtypes(include=np.number).columns
    cols_to_scale = [col for col in numerical_cols if col != target_col]
    
    scaler.fit(train_df[cols_to_scale])
    train_normalized[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
    test_normalized[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
    
    logger.info(f"Scaled {len(cols_to_scale)} numerical columns: {list(cols_to_scale)}")
    mlflow.sklearn.log_model(scaler, "min_max_scaler")
    return train_normalized, test_normalized, scaler


def encode_categorical_data(train_df, test_df, logger, params):
    """Encodes categorical data, logs column list, and aligns dataframes."""
    logger.info("Starting categorical data encoding.")

    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns
    logger.info(f"Found categorical columns: {list(categorical_cols)}")
    
    train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True, dtype=int)
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True, dtype=int)
    
    encoded_columns = train_encoded.columns.tolist()
    logger.info(f"Encoded {len(categorical_cols)} categorical columns, resulting in {len(encoded_columns)} total features")
    
    # Log the columns list as a JSON artifact
    mlflow.log_dict({"encoded_columns": encoded_columns}, "encoded_columns_info")

    train_aligned, test_aligned = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)
    test_aligned = test_aligned[encoded_columns]

    return train_aligned, test_aligned, encoded_columns


def save_artifact_locally(artifact, artifact_path, artifact_type="model", logger=None):
    """Saves artifact (model or data) locally."""
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    
    if artifact_type == "model":
        joblib.dump(artifact, artifact_path)
    elif artifact_type == "json":
        with open(artifact_path, 'w') as f:
            json.dump(artifact, f, indent=4)
    elif artifact_type == "text":
        with open(artifact_path, 'w') as f:
            f.write('\n'.join(artifact) if isinstance(artifact, list) else str(artifact))
    
    if logger:
        logger.info(f"{artifact_type.capitalize()} saved locally to {artifact_path}")


def upload_to_s3(local_path, s3_bucket, s3_key, logger, artifact_name="artifact"):
    """Uploads file to AWS S3."""
    try:
        region_name = os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')
        s3_client = boto3.client('s3', region_name=region_name)
        
        logger.info(f"Uploading {artifact_name} to S3: s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(local_path, s3_bucket, s3_key)
        logger.info(f"{artifact_name} successfully uploaded to S3")
        
        # Log S3 path to MLflow
        mlflow.log_param(f"{artifact_name}_s3_path", f"s3://{s3_bucket}/{s3_key}")
        return True
        
    except ClientError as e:
        logger.error(f"AWS error uploading {artifact_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error uploading {artifact_name} to S3: {e}")
        return False


if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    target_col = params['base']['target_col']
    
    # Set up MLflow under parent experiment
    stage = setup_mlflow_experiment("feature_engineering")
    
    # Start run with stage-specific name and tags
    with mlflow.start_run(run_name="feature_engineering"):
        mlflow.set_tag("stage", stage)
        mlflow.set_tag("pipeline_step", "3_feature_engineering")
        
        try:
            # Load processed data
            train_path = os.path.join(params['data']['processed_dir'], params['data']['train_processed_csv'])
            test_path = os.path.join(params['data']['processed_dir'], params['data']['test_processed_csv'])
            
            logger.info(f"Loading processed data from {train_path} and {test_path}")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            logger.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            mlflow.log_param("train_shape", str(train_data.shape))
            mlflow.log_param("test_shape", str(test_data.shape))

            # Normalization with scaler saved
            logger.info("Normalizing numerical features...")
            norm_train, norm_test, scaler = normalize_data(train_data, test_data, target_col, logger, params)
            
            # Encoding with columns saved
            logger.info("Encoding categorical features...")
            eng_train, eng_test, encoded_columns = encode_categorical_data(norm_train, norm_test, logger, params)

            # Setup directories
            featured_dir = params['data']['featured_dir']
            artifacts_dir = params['data']['artifacts_dir']
            os.makedirs(featured_dir, exist_ok=True)
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Save engineered data locally
            train_featured_path = os.path.join(featured_dir, params['data']['train_engineered_csv'])
            test_featured_path = os.path.join(featured_dir, params['data']['test_engineered_csv'])
            
            eng_train.to_csv(train_featured_path, index=False)
            eng_test.to_csv(test_featured_path, index=False)
            logger.info(f"Engineered data saved to {featured_dir}")

            mlflow.log_artifact(train_featured_path, "featured_data")
            mlflow.log_artifact(test_featured_path, "featured_data")
            
            # Save artifacts locally
            scaler_path = os.path.join(artifacts_dir, params['artifacts']['scaler'])
            encoded_cols_path = os.path.join(artifacts_dir, params['artifacts']['encoded_columns'])
            
            save_artifact_locally(scaler, scaler_path, artifact_type="model", logger=logger)
            save_artifact_locally({"encoded_columns": encoded_columns}, encoded_cols_path, artifact_type="json", logger=logger)
            
            logger.info("Uploading feature engineering artifacts to S3...")
            
            # Get S3 configuration
            s3_bucket = params['data']['s3_bucket']
            s3_prefix = "feature_engineering"
            
            # Upload scaler to S3
            scaler_s3_key = f"{s3_prefix}/{params['artifacts']['scaler']}"
            scaler_upload = upload_to_s3(scaler_path, s3_bucket, scaler_s3_key, logger, artifact_name="scaler")
            
            # Upload encoded columns to S3
            encoded_cols_s3_key = f"{s3_prefix}/{params['artifacts']['encoded_columns']}"
            encoded_cols_upload = upload_to_s3(encoded_cols_path, s3_bucket, encoded_cols_s3_key, logger, artifact_name="encoded_columns")
            
            # Upload engineered datasets to S3
            train_featured_s3_key = f"{s3_prefix}/{params['data']['train_engineered_csv']}"
            test_featured_s3_key = f"{s3_prefix}/{params['data']['test_engineered_csv']}"
            train_featured_upload = upload_to_s3(train_featured_path, s3_bucket, train_featured_s3_key, logger, artifact_name="train_engineered_data")
            test_featured_upload = upload_to_s3(test_featured_path, s3_bucket, test_featured_s3_key, logger, artifact_name="test_engineered_data")
            
            # Log upload status to MLflow
            mlflow.log_param("scaler_s3_upload", str(scaler_upload))
            mlflow.log_param("encoded_columns_s3_upload", str(encoded_cols_upload))
            mlflow.log_param("train_featured_s3_upload", str(train_featured_upload))
            mlflow.log_param("test_featured_s3_upload", str(test_featured_upload))
            
            # Log artifact locations
            mlflow.log_param("artifacts_dir", artifacts_dir)
            mlflow.log_param("featured_dir", featured_dir)
            
            logger.info("Feature engineering and S3 uploads complete.")
            logger.info("All artifacts ready for model training stage.")
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}", exc_info=True)
            mlflow.log_param("status", "failed")
            raise