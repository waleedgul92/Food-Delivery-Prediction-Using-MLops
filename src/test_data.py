# src/test_data.py
import os
import pandas as pd
import json
import joblib
import logging
import yaml
import boto3
from botocore.exceptions import ClientError
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
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
            logging.FileHandler('log_data/model_evaluation.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_params():
    """Loads parameters from params.yaml"""
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)


def download_from_s3(s3_bucket, s3_key, local_path, logger, item_name="file"):
    """Downloads a file from S3 to local storage."""
    try:
        region_name = os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')
        s3_client = boto3.client('s3', region_name=region_name)
        
        logger.info(f"Downloading {item_name} from S3: s3://{s3_bucket}/{s3_key}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(s3_bucket, s3_key, local_path)
        logger.info(f"{item_name} successfully downloaded from S3 to {local_path}")
        return True
        
    except ClientError as e:
        logger.error(f"AWS error downloading {item_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error downloading {item_name} from S3: {e}")
        return False


def load_model_from_local(model_path, logger):
    """Loads model from local storage."""
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None


def load_scaler_from_local(scaler_path, logger):
    """Loads scaler from local storage."""
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler from {scaler_path}: {e}")
        return None


def load_encoded_columns(encoded_cols_path, logger):
    """Loads encoded columns mapping from JSON."""
    try:
        with open(encoded_cols_path, 'r') as f:
            data = json.load(f)
        encoded_columns = data.get('encoded_columns', [])
        logger.info(f"Loaded {len(encoded_columns)} encoded columns from {encoded_cols_path}")
        return encoded_columns
    except Exception as e:
        logger.error(f"Error loading encoded columns from {encoded_cols_path}: {e}")
        return None


def evaluate_model(params, logger):
    """
    Downloads model, scaler, and test data from S3, then evaluates the model.
    """
    with mlflow.start_run(run_name="champion_model_evaluation"):
        try:
            logger.info("--- Starting Champion Model Evaluation ---")
            
            # Get S3 configuration
            s3_bucket = params['data']['s3_bucket']
            
            # Setup local temporary directory for downloads
            temp_dir = 'temp_s3_downloads'
            os.makedirs(temp_dir, exist_ok=True)
            
            logger.info("=== Step 1: Downloading artifacts from S3 ===")
            
            # Download model from S3
            model_filename = params['train']['model_name']  # artifacts/best_delivery_time_model.joblib
            model_s3_key = f"models/{model_filename}"
            local_model_path = os.path.join(temp_dir, os.path.basename(model_filename))
            model_downloaded = download_from_s3(s3_bucket, model_s3_key, local_model_path, logger, item_name="trained_model")
            
            if not model_downloaded:
                logger.error("Failed to download model from S3. Aborting evaluation.")
                mlflow.log_param("status", "failed_model_download")
                return
            
            # Download scaler from S3
            scaler_filename = params['artifacts']['scaler']
            scaler_s3_key = f"feature_engineering/{scaler_filename}"
            local_scaler_path = os.path.join(temp_dir, scaler_filename)
            scaler_downloaded = download_from_s3(s3_bucket, scaler_s3_key, local_scaler_path, logger, item_name="scaler")
            
            if not scaler_downloaded:
                logger.warning("Scaler download failed, but continuing without scaling...")
            
            # Download encoded columns from S3
            encoded_cols_filename = params['artifacts']['encoded_columns']
            encoded_cols_s3_key = f"feature_engineering/{encoded_cols_filename}"
            local_encoded_cols_path = os.path.join(temp_dir, encoded_cols_filename)
            encoded_cols_downloaded = download_from_s3(s3_bucket, encoded_cols_s3_key, local_encoded_cols_path, logger, item_name="encoded_columns")
            
            if not encoded_cols_downloaded:
                logger.warning("Encoded columns download failed, but continuing...")
            
            # Download test engineered data from S3
            test_data_s3_key = f"feature_engineering/{params['data']['test_engineered_csv']}"
            local_test_data_path = os.path.join(temp_dir, params['data']['test_engineered_csv'])
            test_data_downloaded = download_from_s3(s3_bucket, test_data_s3_key, local_test_data_path, logger, item_name="test_data")
            
            if not test_data_downloaded:
                logger.error("Failed to download test data from S3. Aborting evaluation.")
                mlflow.log_param("status", "failed_test_data_download")
                return
            
            logger.info("=== Step 2: Loading artifacts ===")
            
            # Load model
            model = load_model_from_local(local_model_path, logger)
            if model is None:
                logger.error("Failed to load model. Aborting evaluation.")
                mlflow.log_param("status", "failed_model_load")
                return
            
            # Load scaler (optional)
            scaler = None
            if scaler_downloaded:
                scaler = load_scaler_from_local(local_scaler_path, logger)
            
            # Load encoded columns (optional)
            encoded_columns = None
            if encoded_cols_downloaded:
                encoded_columns = load_encoded_columns(local_encoded_cols_path, logger)
            
            logger.info("=== Step 3: Loading and preparing test data ===")
            
            # Load test data
            test_df = pd.read_csv(local_test_data_path)
            target_col = params['base']['target_col']
            
            X_test = test_df.drop(columns=[target_col], errors='ignore')
            y_test = test_df[target_col]
            
            logger.info(f"Test data loaded: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("test_features", X_test.shape[1])
            
            logger.info("=== Step 4: Making predictions ===")
            
            # Make predictions
            predictions = model.predict(X_test)
            logger.info(f"Predictions generated for {len(predictions)} samples")
            
            logger.info("=== Step 5: Evaluating predictions ===")
            
            # Calculate metrics
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            logger.info(f"--- Model Evaluation Metrics ---")
            logger.info(f"R² Score:  {r2:.4f}")
            logger.info(f"MSE:       {mse:.4f}")
            logger.info(f"RMSE:      {rmse:.4f}")
            logger.info(f"MAE:       {mae:.4f}")
            logger.info(f"MAPE:      {mape:.4f}")
            
            # Log metrics to MLflow
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mape", mape)
            
            # Log artifact sources
            mlflow.log_param("model_s3_source", f"s3://{s3_bucket}/{model_s3_key}")
            mlflow.log_param("test_data_s3_source", f"s3://{s3_bucket}/{test_data_s3_key}")
            
            logger.info("=== Step 6: Saving predictions ===")
            
            # Save predictions to CSV
            prediction_df = pd.DataFrame({
                'Actual_Time': y_test.values,
                'Predicted_Time': predictions,
                'Error': y_test.values - predictions,
                'Absolute_Error': abs(y_test.values - predictions)
            })
            
            results_dir = os.path.dirname(params['evaluate']['predictions_csv'])
            os.makedirs(results_dir, exist_ok=True)
            prediction_df.to_csv(params['evaluate']['predictions_csv'], index=False)
            logger.info(f"Predictions saved to {params['evaluate']['predictions_csv']}")
            
            # Save evaluation metrics to JSON
            metrics_file = os.path.join(results_dir, 'evaluation_metrics.json')
            metrics = {
                "r2_score": float(r2),
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "test_samples": int(X_test.shape[0]),
                "test_features": int(X_test.shape[1])
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_file}")
            
            # Log artifacts to MLflow
            mlflow.log_artifact(params['evaluate']['predictions_csv'], "predictions")
            mlflow.log_artifact(metrics_file, "metrics")
            
            logger.info("=== Model Evaluation Complete ===")
            mlflow.log_param("status", "success")
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}", exc_info=True)
            mlflow.log_param("status", "failed")
            raise


if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    
    # Set up MLflow experiment under "Food Delivery System" hierarchy
    experiment_name = setup_mlflow_experiment("test_model")
    logger.info(f"Using MLflow experiment: {experiment_name}")
    
    evaluate_model(params, logger)