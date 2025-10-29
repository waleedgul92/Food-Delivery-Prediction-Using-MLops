# src/train_data.py
import os
import pandas as pd
import optuna
import logging
import yaml
import json
import joblib
import boto3
from botocore.exceptions import ClientError
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from .mlflow_config import setup_mlflow_experiment

load_dotenv()


def setup_logging():
    """Configures logging."""
    os.makedirs('log_data', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('log_data/model_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_params():
    """Loads parameters from params.yaml"""
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)


def load_featured_data(params, logger):
    """Loads featured data from featured directory."""
    target_col = params['base']['target_col']
    train_path = os.path.join(params['data']['featured_dir'], params['data']['train_engineered_csv'])
    test_path = os.path.join(params['data']['featured_dir'], params['data']['test_engineered_csv'])
    
    logger.info(f"Loading featured data from {train_path} and {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[target_col], errors='ignore')
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col], errors='ignore')
    y_test = test_df[target_col]
    
    logger.info(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_train, y_train, X_test, y_test


def get_models(params):
    """Returns a dictionary of baseline models."""
    random_state = params['base']['random_state']
    baseline_params = params.get('baseline_params', {})
    
    return {
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=random_state),
        "RandomForestRegressor": RandomForestRegressor(
            random_state=random_state,
            **baseline_params.get('RandomForestRegressor', {})
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
        "XGBRegressor": xgb.XGBRegressor(
            random_state=random_state,
            **baseline_params.get('XGBRegressor', {})
        )
    }


def train_and_evaluate(model, X_train, y_train, X_test, y_test, logger):
    """Trains model and evaluates on test set."""
    logger.info(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    
    logger.info(f"{model.__class__.__name__} - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return r2, mse, rmse, mae


def objective(trial, X_train, y_train, model_name, params, logger):
    """Optuna objective function for hyperparameter optimization."""
    hpo_params = params.get('train', {}).get('hpo', {}).get(model_name, {})
    trial_params = {}
    
    for param_name, param_config in hpo_params.items():
        param_type = param_config.get('type', 'float')
        args = param_config.get('args', [])
        log_scale = param_config.get('log', False)
        
        if param_type == 'int':
            trial_params[param_name] = trial.suggest_int(param_name, *args)
        elif param_type == 'float':
            trial_params[param_name] = trial.suggest_float(param_name, *args, log=log_scale)
        elif param_type == 'categorical':
            trial_params[param_name] = trial.suggest_categorical(param_name, args)

    trial_params['random_state'] = params['base']['random_state']
    model_class = get_models(params)[model_name].__class__
    model = model_class(**trial_params)
    
    cv_folds = params.get('train', {}).get('cv_folds', 3)
    score = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2').mean()
    return score


def save_model_locally(model, model_path, logger):
    """Saves model locally using joblib."""
    # Normalize path for Windows/Unix compatibility
    model_path = model_path.replace('\\', '/')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved locally to {model_path}")


def upload_to_s3(local_path, s3_bucket, s3_key, logger, artifact_name="model"):
    """Uploads file to AWS S3."""
    try:
        region_name = os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')
        s3_client = boto3.client('s3', region_name=region_name)
        
        logger.info(f"Uploading {artifact_name} to S3: s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(local_path, s3_bucket, s3_key)
        logger.info(f"{artifact_name} successfully uploaded to S3")
        
        mlflow.log_param(f"{artifact_name}_s3_path", f"s3://{s3_bucket}/{s3_key}")
        return True
        
    except ClientError as e:
        logger.error(f"AWS error uploading {artifact_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error uploading {artifact_name} to S3: {e}")
        return False


def save_metrics_to_file(metrics, file_path, logger):
    """Saves metrics to JSON file."""
    # Normalize path separators
    file_path = file_path.replace('\\', '/')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {file_path}")


def manage_model_registry(model_name, model_info, challenger_r2, logger):
    """Manages model registry."""
    logger.info(f"Managing model registry for '{model_name}'...")
    client = MlflowClient()
    
    try:
        client.create_registered_model(model_name)
        logger.info(f"Created new registered model: {model_name}")
    except mlflow.exceptions.RestException:
        logger.info(f"Model '{model_name}' already exists in registry")
    
    version = client.create_model_version(
        name=model_name,
        source=model_info.model_uri,
        run_id=model_info.run_id,
        tags={"r2_score": str(challenger_r2)}
    )
    
    logger.info(f"Registered model version {version.version} with R2: {challenger_r2:.4f}")


if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()

    # Set up MLflow experiment
    experiment_name = setup_mlflow_experiment("train_model")
    logger.info(f"Using MLflow experiment: {experiment_name}")

    # Load data
    X_train, y_train, X_test, y_test = load_featured_data(params, logger)
    models = get_models(params)

    with mlflow.start_run(run_name="model_training_and_evaluation"):
        try:
            logger.info("--- Baseline Model Evaluation ---")
            scores = {}

            for name, model in models.items():
                with mlflow.start_run(run_name=f"baseline_{name}", nested=True):
                    r2, mse, rmse, mae = train_and_evaluate(model, X_train, y_train, X_test, y_test, logger)
                    scores[name] = r2
                    mlflow.log_params(model.get_params())
                    mlflow.log_metric("r2_score", r2)
                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("mae", mae)

            best_model_name = max(scores, key=scores.get)
            best_r2 = scores[best_model_name]
            mlflow.set_tag("best_baseline_model", best_model_name)
            logger.info(f"Best baseline model: {best_model_name} with R2: {best_r2:.4f}")

            logger.info(f"--- Hyperparameter Tuning: {best_model_name} ---")
            n_trials = params.get('train', {}).get('n_trials', 10)
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=params['base']['random_state'])
            )
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, best_model_name, params, logger),
                n_trials=n_trials
            )

            logger.info(f"Best hyperparameters: {study.best_params}")
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_hpo_r2", study.best_value)

            logger.info("--- Training Final Model with Best Hyperparameters ---")
            final_model_class = models[best_model_name].__class__
            final_model = final_model_class(**study.best_params, random_state=params['base']['random_state'])

            final_r2, final_mse, final_rmse, final_mae = train_and_evaluate(final_model, X_train, y_train, X_test, y_test, logger)

            mlflow.log_metric("final_r2_score", final_r2)
            mlflow.log_metric("final_mse", final_mse)
            mlflow.log_metric("final_rmse", final_rmse)
            mlflow.log_metric("final_mae", final_mae)

            # Save model locally
            artifacts_dir = params['data']['artifacts_dir']
            model_filename = os.path.basename(params['train']['model_name'])  # Get just filename
            model_path = os.path.join(artifacts_dir, model_filename)
            # Normalize path for consistency
            model_path = model_path.replace('\\', '/')
            save_model_locally(final_model, model_path, logger)
            
            # Log model to MLflow
            model_info = mlflow.sklearn.log_model(sk_model=final_model, artifact_path="best_model")
            logger.info(f"Best model '{best_model_name}' logged to MLflow")

            # Upload model to S3
            s3_bucket = params['data']['s3_bucket']
            s3_model_key = f"models/{model_filename}"
            model_upload = upload_to_s3(model_path, s3_bucket, s3_model_key, logger, artifact_name="trained_model")
            mlflow.log_param("model_s3_upload", str(model_upload))
            
            # Save and upload training metrics
            metrics = {
                "model": best_model_name,
                "final_r2_score": float(final_r2),
                "final_mse": float(final_mse),
                "final_rmse": float(final_rmse),
                "final_mae": float(final_mae),
                "best_hyperparameters": study.best_params
            }
            
            metrics_path = os.path.join(artifacts_dir, "training_metrics.json")
            # Normalize path separators
            metrics_path = metrics_path.replace('\\', '/')
            save_metrics_to_file(metrics, metrics_path, logger)
            mlflow.log_artifact(metrics_path, "metrics")
            
            # Register model in MLflow
            model_name = params['model_registry']['name']
            manage_model_registry(model_name=model_name, model_info=model_info, challenger_r2=final_r2, logger=logger)
            
            logger.info("Model training and upload complete.")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            mlflow.log_param("status", "failed")
            raise