import os
import pandas as pd
import logging
import yaml
from sklearn.metrics import r2_score, mean_squared_error
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('log_data/model_evaluation.log', mode='w'), logging.StreamHandler()])
    return logging.getLogger(__name__)

def load_params():
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)

def evaluate_model(params, logger):
    """
    Loads the latest model from MLflow, evaluates it on the test set,
    and logs the results.
    """
    with mlflow.start_run(run_name="final_model_evaluation"):
        logger.info("--- Starting Final Model Evaluation on Test Data ---")
        
        # Search for the main training run (not nested runs)
        # Look for runs with the run_name "model_training_and_evaluation"
        try:
            training_runs = mlflow.search_runs(
                experiment_names=["Default"],
                filter_string="tags.mlflow.runName = 'model_training_and_evaluation'",
                order_by=["start_time DESC"],
                max_results=1
            )
        except:
            # Fallback 1: Search for runs that have final_r2_score metric (unique to your training)
            try:
                training_runs = mlflow.search_runs(
                    experiment_names=["Default"],
                    filter_string="metrics.final_r2_score >= 0",
                    order_by=["start_time DESC"],
                    max_results=1
                )
            except:
                # Fallback 2: Search for runs that have logged a model artifact
                try:
                    training_runs = mlflow.search_runs(
                        experiment_names=["Default"],
                        filter_string="status = 'FINISHED'",
                        order_by=["start_time DESC"],
                        max_results=10  # Get more runs to filter
                    )
                    
                    # Filter runs that have a model artifact
                    training_runs_with_model = []
                    for idx, run in training_runs.iterrows():
                        run_id = run.run_id
                        try:
                            # Check if model artifact exists
                            artifacts = mlflow.tracking.MlflowClient().list_artifacts(run_id)
                            if any(artifact.path == 'best_model' for artifact in artifacts):
                                training_runs_with_model.append(run)
                                break
                        except:
                            continue
                    
                    if training_runs_with_model:
                        training_runs = pd.DataFrame([training_runs_with_model[0]]).reset_index(drop=True)
                    else:
                        training_runs = pd.DataFrame()
                        
                except Exception as e:
                    logger.error(f"Error searching for training runs: {e}")
                    # Final fallback - get the most recent completed run
                    training_runs = mlflow.search_runs(
                        experiment_names=["Default"],
                        order_by=["start_time DESC"],
                        max_results=1
                    )
        
        if len(training_runs) == 0:
            logger.error("No training run found. Please run the train_model stage first.")
            return
            
        training_run_id = training_runs.iloc[0].run_id
        model_uri = f"runs:/{training_run_id}/best_model"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from run ID: {training_run_id}")
        except Exception as e:
            logger.error(f"Failed to load model from URI: {model_uri}. Error: {e}")
            return

        test_data_path = os.path.join(params['data']['featured_dir'], params['data']['test_engineered_csv'])
        target_col = params['base']['target_col']
        
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        predictions = model.predict(X_test)
        
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        
        mlflow.log_metric("test_r2_score", r2)
        mlflow.log_metric("test_mse", mse)
        
        logger.info(f"--- Final Model Performance --- R2: {r2:.4f} | MSE: {mse:.4f}")
        
        prediction_df = pd.DataFrame({'Actual_Time': y_test, 'Predicted_Time': predictions})
        predictions_path = params['evaluate']['predictions_csv']
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        prediction_df.to_csv(predictions_path, index=False)

        mlflow.log_artifact(predictions_path, "predictions")
        logger.info(f"Saved predictions to '{predictions_path}'")

if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    evaluate_model(params, logger)