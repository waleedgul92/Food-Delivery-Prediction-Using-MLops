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
import joblib

def setup_logging():
    """Configures logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('log_data/data_processing.log'), logging.StreamHandler()])
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
        'Distance_km': 'Distance', 'Traffic_Level': 'Traffic_Level', 'Vehicle_Type': 'Vehicle_Type',
        'Preparation_Time_min': 'Preparation_Time', 'Courier_Experience_yrs': 'Courier_Experience',
        'Delivery_Time_min': 'Delivery_Time_Minutes'
    }
    dataset.rename(columns=column_rename_map, inplace=True)
    return dataset

def impute_missing_data(train_df, test_df, logger, params):
    """Imputes missing data and saves the imputers."""
    logger.info("Starting missing data imputation.")
    train_imputed, test_imputed = train_df.copy(), test_df.copy()
    artifacts_dir = params['data']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)

    categorical_cols = train_imputed.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        train_imputed[categorical_cols] = imputer_cat.fit_transform(train_imputed[categorical_cols])
        test_imputed[categorical_cols] = imputer_cat.transform(test_imputed[categorical_cols])
        joblib.dump(imputer_cat, os.path.join(artifacts_dir, params['artifacts']['cat_imputer']))

    numerical_cols = train_imputed.select_dtypes(include=np.number).columns
    if not numerical_cols.empty:
        imputer_num = IterativeImputer(random_state=params['base']['random_state'])
        train_imputed[numerical_cols] = imputer_num.fit_transform(train_imputed[numerical_cols])
        test_imputed[numerical_cols] = imputer_num.transform(test_imputed[numerical_cols])
        joblib.dump(imputer_num, os.path.join(artifacts_dir, params['artifacts']['num_imputer']))
    
    return train_imputed, test_imputed

def remove_outliers_isolation_forest(df, logger, params):
    """Removes outliers using Isolation Forest."""
    logger.info("Starting outlier removal.")
    df_out = df.copy()
    numerical_cols = df_out.select_dtypes(include=np.number).columns
    iso_forest = IsolationForest(contamination=params['data_process']['contamination'], random_state=params['base']['random_state'])
    outlier_pred = iso_forest.fit_predict(df_out[numerical_cols])
    outlier_mask = outlier_pred == -1
    logger.info(f"Found and removed {sum(outlier_mask)} outliers.")
    return df_out[~outlier_mask]

def save_eda_plots(df, output_dir, logger):
    """Generates and saves EDA plots."""
    logger.info(f"Generating and saving EDA plots to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    # ... (Plotting logic remains the same)
    plt.close('all') # Close all figures to free memory

if __name__ == "__main__":
    logger = setup_logging()
    params = load_params()
    
    train_data_path = os.path.join(params['data']['interim_dir'], params['data']['train_csv'])
    test_data_path = os.path.join(params['data']['interim_dir'], params['data']['test_csv'])
    processed_output_dir = params['data']['processed_dir']
    os.makedirs(processed_output_dir, exist_ok=True)
    
    train_data = load_and_rename_data(train_data_path, logger)
    test_data = load_and_rename_data(test_data_path, logger)
    
    imputed_train, imputed_test = impute_missing_data(train_data, test_data, logger, params)
    save_eda_plots(imputed_train, params['data']['viz_dir'], logger)
    final_train = remove_outliers_isolation_forest(imputed_train, logger, params)
    
    final_train.to_csv(os.path.join(processed_output_dir, params['data']['train_processed_csv']), index=False)
    imputed_test.to_csv(os.path.join(processed_output_dir, params['data']['test_processed_csv']), index=False)
    logger.info("Data processing pipeline complete.")
