# src/feature_engineering.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import yaml
import joblib
import json
import mlflow
import dagshub
dagshub.init(repo_owner='hwaleed0035', repo_name='Food-Delivery-Prediction-Using-MLops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/hwaleed0035/Food-Delivery-Prediction-Using-MLops.mlflow")
mlflow.set_experiment("Food Delivery Prediction Model Training")


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('log_data/feature_engineering.log'), logging.StreamHandler()])
    return logging.getLogger(__name__)

def load_params():
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
    
    # Log the scaler to MLflow
    mlflow.sklearn.log_model(scaler, "min_max_scaler")
    return train_normalized, test_normalized

def encode_categorical_data(train_df, test_df, logger, params):
    """Encodes categorical data, logs column list, and aligns dataframes."""
    logger.info("Starting categorical data encoding.")

    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns
    train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True, dtype=int)
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True, dtype=int)
    
    encoded_columns = train_encoded.columns.tolist()
    # Log the columns list as a JSON artifact
    mlflow.log_dict(encoded_columns, "encoded_columns.json")

    train_aligned, test_aligned = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)
    test_aligned = test_aligned[encoded_columns]

    return train_aligned, test_aligned

if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    target_col = params['base']['target_col']
    
    with mlflow.start_run(run_name="feature_engineering", nested=True):
        train_path = os.path.join(params['data']['processed_dir'], params['data']['train_processed_csv'])
        test_path = os.path.join(params['data']['processed_dir'], params['data']['test_processed_csv'])
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        norm_train, norm_test = normalize_data(train_data, test_data, target_col, logger, params)
        eng_train, eng_test = encode_categorical_data(norm_train, norm_test, logger, params)

        engineered_output_dir = params['data']['featured_dir']
        os.makedirs(engineered_output_dir, exist_ok=True)
        
        train_engineered_path = os.path.join(engineered_output_dir, params['data']['train_engineered_csv'])
        test_engineered_path = os.path.join(engineered_output_dir, params['data']['test_engineered_csv'])
        
        eng_train.to_csv(train_engineered_path, index=False)
        eng_test.to_csv(test_engineered_path, index=False)

        mlflow.log_artifact(train_engineered_path, "featured_data")
        mlflow.log_artifact(test_engineered_path, "featured_data")
        
        logger.info("Feature engineering complete.")