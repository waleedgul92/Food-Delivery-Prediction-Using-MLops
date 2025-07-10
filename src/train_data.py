# src/train_data.py
import os
import pandas as pd
import optuna
import joblib
import logging
import yaml
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('log_data/model_training.log', mode='w'), logging.StreamHandler()])
    return logging.getLogger(__name__)

def load_params():
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)

def load_featured_data(params, logger):
    target_col = params['base']['target_col']
    train_path = os.path.join(params['data']['featured_dir'], params['data']['train_engineered_csv'])
    test_path = os.path.join(params['data']['featured_dir'], params['data']['test_engineered_csv'])
    logger.info(f"Loading data from {train_path} and {test_path}")
    X_train = pd.read_csv(train_path).drop(columns=[target_col], errors='ignore')
    y_train = pd.read_csv(train_path)[target_col]
    X_test = pd.read_csv(test_path).drop(columns=[target_col], errors='ignore')
    y_test = pd.read_csv(test_path)[target_col]
    return X_train, y_train, X_test, y_test

def get_models(params):
    random_state = params['base']['random_state']
    return {
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=random_state),
        "RandomForestRegressor": RandomForestRegressor(random_state=random_state, **params['baseline_params']['RandomForestRegressor']),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
        "XGBRegressor": xgb.XGBRegressor(random_state=random_state, **params['baseline_params'].get('XGBRegressor', {}))
    }

def train_and_evaluate(model, X_train, y_train, X_test, y_test, logger):
    model_name = model.__class__.__name__
    logger.info(f"Training and evaluating {model_name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    logger.info(f"--- {model_name} --- R2: {r2:.4f} | MSE: {mse:.4f}")
    return r2

def objective(trial, X_train, y_train, model_name, params):
    hpo_params = params['train']['hpo'][model_name]
    trial_params = {}
    for param_name, param_config in hpo_params.items():
        param_type = param_config['type']
        args = param_config['args']
        if param_type == 'int':
            trial_params[param_name] = trial.suggest_int(param_name, *args)
        elif param_type == 'float':
            trial_params[param_name] = trial.suggest_float(param_name, *args, log=param_config.get('log', False))
        elif param_type == 'categorical':
            trial_params[param_name] = trial.suggest_categorical(param_name, args)

    trial_params['random_state'] = params['base']['random_state']
    model_class = get_models(params)[model_name].__class__
    model = model_class(**trial_params)
    score = cross_val_score(model, X_train, y_train, cv=params['train']['cv_folds'], scoring='neg_mean_squared_error').mean()
    return score

if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    try:
        X_train, y_train, X_test, y_test = load_featured_data(params, logger)
        models = get_models(params)
        
        logger.info("--- Starting Baseline Model Evaluation ---")
        scores = {name: train_and_evaluate(model, X_train, y_train, X_test, y_test, logger) for name, model in models.items()}
        
        best_model_name = max(scores, key=scores.get)
        logger.info(f"\n--- Best Baseline Model (by R2): {best_model_name} ---")
        
        logger.info(f"\n--- Performing Hyperparameter Tuning for {best_model_name} based on MSE ---")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=params['base']['random_state']))
        study.optimize(lambda trial: objective(trial, X_train, y_train, best_model_name, params), n_trials=params['train']['n_trials'])
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        logger.info("\n--- Training Final Model with Best Hyperparameters ---")
        final_model_class = models[best_model_name].__class__
        final_model = final_model_class(**study.best_params, random_state=params['base']['random_state'])
        train_and_evaluate(final_model, X_train, y_train, X_test, y_test, logger)
        
        model_path = params['train']['model_name']
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        joblib.dump(final_model, model_path)
        logger.info(f"Best model saved to '{model_path}'")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
