# src/test_data.py
import os
import pandas as pd
import joblib
import logging
import yaml
from sklearn.metrics import r2_score, mean_squared_error

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('log_data/model_evaluation.log', mode='w'), logging.StreamHandler()])
    return logging.getLogger(__name__)

def load_params():
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)

def evaluate_model(params, logger):
    logger.info("--- Starting Final Model Evaluation on Test Data ---")
    test_data_path = os.path.join(params['data']['featured_dir'], params['data']['test_engineered_csv'])
    model_path = params['train']['model_name']
    target_col = params['base']['target_col']
    
    test_df = pd.read_csv(test_data_path)
    model = joblib.load(model_path)
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    logger.info(f"--- Final Model Performance --- R2: {r2:.4f} | MSE: {mse:.4f}")
    
    prediction_df = pd.DataFrame({'Actual_Time': y_test, 'Predicted_Time': predictions})
    predictions_path = params['evaluate']['predictions_csv']
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    prediction_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved predictions to '{predictions_path}'")

if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    evaluate_model(params, logger)