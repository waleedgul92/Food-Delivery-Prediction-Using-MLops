import os
import pandas as pd
import optuna
import logging
import yaml
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import dagshub
from app.model_handler import setup_logging, train_and_evaluate, manage_model_registry
dagshub.init(repo_owner='hwaleed0035', repo_name='Food-Delivery-Prediction-Using-MLops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/hwaleed0035/Food-Delivery-Prediction-Using-MLops.mlflow")
mlflow.set_experiment("Food Delivery Prediction Model Training")



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

    # Load data
    X_train, y_train, X_test, y_test = load_featured_data(params, logger)
    models = get_models(params)

    with mlflow.start_run(run_name="model_training_and_evaluation") as parent_run:
        logger.info("--- Baseline Model Evaluation ---")
        scores = {}

        for name, model in models.items():
            with mlflow.start_run(run_name=f"baseline_{name}", nested=True):
                r2, mse = train_and_evaluate(model, X_train, y_train, X_test, y_test, logger)
                scores[name] = r2
                mlflow.log_params(model.get_params())
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mse", mse)

        best_model_name = max(scores, key=scores.get)
        mlflow.set_tag("best_baseline_model", best_model_name)

        logger.info(f"--- Hyperparameter Tuning: {best_model_name} ---")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=params['base']['random_state'])
        )
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, best_model_name, params),
            n_trials=params['train']['n_trials']
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_hpo_mse", study.best_value)

        logger.info("--- Training Final Model with Best Hyperparameters ---")
        final_model_class = models[best_model_name].__class__
        final_model = final_model_class(**study.best_params, random_state=params['base']['random_state'])

        final_r2, final_mse = train_and_evaluate(final_model, X_train, y_train, X_test, y_test, logger)

        mlflow.log_metric("final_r2_score", final_r2)
        mlflow.log_metric("final_mse", final_mse)

        model_info = mlflow.sklearn.log_model(sk_model=final_model, artifact_path="best_model")
        logger.info(f"✅ Best model '{best_model_name}' logged to MLflow")

        model_name = params['model_registry']['name']
        manage_model_registry(model_name=model_name, model_info=model_info, challenger_r2=final_r2, logger=logger)