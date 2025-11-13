# src/data_process.py
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import IsolationForest
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
            logging.FileHandler('log_data/data_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_params():
    """Loads parameters from params.yaml"""
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)


def load_and_rename_data(file_path, logger):
    """Loads data and renames columns."""
    logger.info(f"Loading and renaming data from {file_path}...")
    dataset = pd.read_csv(file_path)
    dataset.drop("Order_ID", axis=1, inplace=True, errors='ignore')
    column_rename_map = {
        'Distance_km': 'Distance',
        'Traffic_Level': 'Traffic_Level',
        'Vehicle_Type': 'Vehicle_Type',
        'Preparation_Time_min': 'Preparation_Time',
        'Courier_Experience_yrs': 'Courier_Experience',
        'Delivery_Time_min': 'Delivery_Time_Minutes'
    }
    dataset.rename(columns=column_rename_map, inplace=True)
    return dataset


def impute_missing_data(train_df, test_df, logger, params):
    """Imputes missing data and logs the imputers to MLflow."""
    logger.info("Starting missing data imputation.")
    train_imputed, test_imputed = train_df.copy(), test_df.copy()

    categorical_cols = train_imputed.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        train_imputed[categorical_cols] = imputer_cat.fit_transform(train_imputed[categorical_cols])
        test_imputed[categorical_cols] = imputer_cat.transform(test_imputed[categorical_cols])
        logger.info(f"Imputed {len(categorical_cols)} categorical columns")
        mlflow.sklearn.log_model(imputer_cat, "categorical_imputer")

    numerical_cols = train_imputed.select_dtypes(include=np.number).columns
    if not numerical_cols.empty:
        imputer_num = IterativeImputer(random_state=params['base']['random_state'])
        train_imputed[numerical_cols] = imputer_num.fit_transform(train_imputed[numerical_cols])
        test_imputed[numerical_cols] = imputer_num.transform(test_imputed[numerical_cols])
        logger.info(f"Imputed {len(numerical_cols)} numerical columns")
        mlflow.sklearn.log_model(imputer_num, "numerical_imputer")
    
    return train_imputed, test_imputed


def remove_outliers_isolation_forest(df, logger, params):
    """Removes outliers using Isolation Forest."""
    logger.info("Starting outlier removal using Isolation Forest.")
    df_out = df.copy()
    numerical_cols = df_out.select_dtypes(include=np.number).columns
    iso_forest = IsolationForest(
        contamination=params['data_process']['contamination'],
        random_state=params['base']['random_state']
    )
    outlier_pred = iso_forest.fit_predict(df_out[numerical_cols])
    outlier_mask = outlier_pred == -1
    num_outliers = sum(outlier_mask)
    logger.info(f"Found and removed {num_outliers} outliers.")
    mlflow.log_metric("outliers_removed", num_outliers)
    return df_out[~outlier_mask]


def save_eda_plots(df, output_dir, logger):
    """Generates and saves EDA plots."""
    logger.info(f"Generating and saving EDA plots to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        numerical_cols = df.select_dtypes(include=np.number).columns
        
        for col in numerical_cols[:5]:
            plt.figure(figsize=(8, 5))
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
            plot_path = os.path.join(output_dir, f'histogram_{col}.png')
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            mlflow.log_artifact(plot_path, "eda_plots")
            plt.close()
        
        logger.info("EDA plots generated and logged to MLflow.")
    except Exception as e:
        logger.warning(f"Could not generate all EDA plots: {e}")
    finally:
        plt.close('all')


if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    
    # Set up MLflow under parent experiment
    stage = setup_mlflow_experiment("data_processing")
    
    # Start run with stage-specific name and tags
    with mlflow.start_run(run_name="data_processing"):
        mlflow.set_tag("stage", stage)
        mlflow.set_tag("pipeline_step", "2_data_processing")
        
        try:
            train_data_path = os.path.join(params['data']['interim_dir'], params['data']['train_csv'])
            test_data_path = os.path.join(params['data']['interim_dir'], params['data']['test_csv'])
            processed_output_dir = params['data']['processed_dir']
            
            os.makedirs(processed_output_dir, exist_ok=True)
            
            logger.info("Loading data...")
            train_data = load_and_rename_data(train_data_path, logger)
            test_data = load_and_rename_data(test_data_path, logger)
            
            logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
            mlflow.log_param("train_data_rows", train_data.shape[0])
            mlflow.log_param("train_data_cols", train_data.shape[1])
            mlflow.log_param("test_data_rows", test_data.shape[0])
            
            logger.info("Imputing missing data...")
            imputed_train, imputed_test = impute_missing_data(train_data, test_data, logger, params)
            
            logger.info("Generating EDA plots...")
            save_eda_plots(imputed_train, params['data']['viz_dir'], logger)
            
            logger.info("Removing outliers...")
            final_train = remove_outliers_isolation_forest(imputed_train, logger, params)
            logger.info(f"Final training data shape after outlier removal: {final_train.shape}")
            
            train_processed_path = os.path.join(processed_output_dir, params['data']['train_processed_csv'])
            test_processed_path = os.path.join(processed_output_dir, params['data']['test_processed_csv'])
            
            final_train.to_csv(train_processed_path, index=False)
            imputed_test.to_csv(test_processed_path, index=False)
            logger.info(f"Processed data saved to {processed_output_dir}")
            
            mlflow.log_artifact(train_processed_path, "processed_data")
            mlflow.log_artifact(test_processed_path, "processed_data")
            
            mlflow.log_param("contamination", params['data_process']['contamination'])
            mlflow.log_param("imputation_strategy_categorical", "most_frequent")
            mlflow.log_param("imputation_strategy_numerical", "iterative")
            
            logger.info("Data processing pipeline complete.")
            
        except Exception as e:
            logger.error(f"Error during data processing: {e}", exc_info=True)
            raise