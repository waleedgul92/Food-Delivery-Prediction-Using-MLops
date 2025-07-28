import os
import pandas as pd
import logging
import yaml
from sklearn.metrics import r2_score, mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# --- DagsHub and MLflow Setup ---
dagshub.init(repo_owner='hwaleed0035', repo_name='Food-Delivery-Prediction-Using-MLops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/hwaleed0035/Food-Delivery-Prediction-Using-MLops.mlflow")
mlflow.set_experiment("Food Delivery Prediction Model Training")


def setup_logging():
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
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)


def get_champion_model(model_name, logger):
    """
    Finds and returns the champion model (highest R2 score) from the registry.
    """
    client = MlflowClient()
    best_version = None
    highest_r2 = -1.0

    try:
        versions = client.get_latest_versions(model_name, stages=["None"])
        if not versions:
            logger.error(f"No versions found for model '{model_name}' in the registry.")
            return None

        for version in versions:
            r2_tag = version.tags.get("r2_score")
            if r2_tag:
                r2_score_val = float(r2_tag)
                if r2_score_val > highest_r2:
                    highest_r2 = r2_score_val
                    best_version = version

        if best_version:
            logger.info(f"Found champion model: Version {best_version.version} with R2 score: {highest_r2:.4f}")
            return mlflow.sklearn.load_model(best_version.source)
        else:
            logger.error("No model version with an 'r2_score' tag found.")
            return None

    except mlflow.exceptions.RestException as e:
        logger.error(f"Could not fetch models from the registry. Error: {e}")
        return None


def evaluate_model(params, logger):
    """
    Loads the champion model from the Model Registry, evaluates it on the test set,
    and logs the results.
    """
    with mlflow.start_run(run_name="champion_model_evaluation"):
        logger.info("--- Starting Champion Model Evaluation on Test Data ---")

        model_name = params['model_registry']['name']
        model = get_champion_model(model_name, logger)

        if model is None:
            logger.error("Could not retrieve the champion model. Aborting evaluation.")
            return

        test_data_path = os.path.join(params['data']['featured_dir'], params['data']['test_engineered_csv'])
        target_col = params['base']['target_col']

        test_df = pd.read_csv(test_data_path)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

        mlflow.log_metric("champion_test_r2_score", r2)
        mlflow.log_metric("champion_test_mse", mse)

        logger.info(f"--- Champion Model Performance --- R2: {r2:.4f} | MSE: {mse:.4f}")

        prediction_df = pd.DataFrame({'Actual_Time': y_test, 'Predicted_Time': predictions})
        predictions_path = params['evaluate']['predictions_csv']
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        prediction_df.to_csv(predictions_path, index=False)

        mlflow.log_artifact(predictions_path, "predictions")
        logger.info(f"Saved champion model predictions to '{predictions_path}'")


if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    evaluate_model(params, logger)
