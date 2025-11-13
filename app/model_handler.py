import os
import time
import pickle
import threading
import pandas as pd
import logging
import mlflow
import mlflow.sklearn
from functools import lru_cache
from datetime import datetime
from mlflow.tracking import MlflowClient
from app.schema import PredictionInput
from src.train_data import load_params, load_featured_data, get_models, train_and_evaluate, manage_model_registry
from mlflow.sklearn import load_model as mlflow_load_model

logger = logging.getLogger(__name__)

# Global variables
model = None
scaler = None
model_loaded = False
model_lock = threading.Lock()
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('log_data/model_training.log', mode='w'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)
@lru_cache(maxsize=1)
def get_cached_model_info():
    try:
        model_name = "DeliveryTimePredictor"
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["None"])
        if not versions:
            return None, None

        best_version = None
        highest_r2 = -1.0
        for version in versions:
            r2 = version.tags.get("r2_score")
            if r2 and float(r2) > highest_r2:
                best_version = version
                highest_r2 = float(r2)
        return best_version, highest_r2

    except Exception as e:
        logger.error(f"Error getting cached model info: {e}")
        return None, None

def load_model_and_scaler():
    global model, scaler, model_loaded

    with model_lock:
        if model_loaded:
            logger.info("🔁 Model already loaded.")
            return True

        try:
            logger.info("🔍 Loading model and scaler...")
            client = MlflowClient()
            best_version, highest_r2 = get_cached_model_info()

            if not best_version:
                logger.error("❌ No valid model found in registry.")
                return False

            logger.info(f"✅ Loading model v{best_version.version} (R2: {highest_r2})")
            model = mlflow.sklearn.load_model(best_version.source)

            # Load scaler (example path, change as needed)
            artifact_path = client.download_artifacts("848ae8ded1dc4771b159b7de93dcb4b9", "min_max_scaler/model.pkl")
            with open(artifact_path, 'rb') as f:
                scaler = pickle.load(f)

            model_loaded = True
            return True

        except Exception as e:
            logger.error(f"❌ Error loading model or scaler: {e}", exc_info=True)
            return False

async def make_prediction(data: PredictionInput):
    global model, scaler
    import asyncio

    if not model or not scaler:
        success = await asyncio.get_event_loop().run_in_executor(None, load_model_and_scaler)
        if not success:
            raise Exception("Model or scaler not loaded")

    try:
        input_data = data.dict()

        feature_order = [
            'Distance', 'Preparation_Time', 'Courier_Experience',
            'Weather_Foggy', 'Weather_Rainy', 'Weather_Snowy', 'Weather_Windy',
            'Traffic_Level_Low', 'Traffic_Level_Medium',
            'Time_of_Day_Evening', 'Time_of_Day_Morning', 'Time_of_Day_Night',
            'Vehicle_Type_Car', 'Vehicle_Type_Scooter'
        ]

        df = pd.DataFrame([[input_data[col] for col in feature_order]], columns=feature_order)
        df[['Distance', 'Preparation_Time', 'Courier_Experience']] = scaler.transform(
            df[['Distance', 'Preparation_Time', 'Courier_Experience']]
        )

        prediction = model.predict(df)[0]
        prediction_id = f"pred_{int(time.time()*1000)}"

        return {
            "predicted_delivery_time": round(prediction, 2),
            "prediction_id": prediction_id
        }

    except Exception as e:
        logger.error("Prediction failed", exc_info=True)
        raise Exception(f"Prediction error: {str(e)}")

def get_model_info():
    try:
        best_version, highest_r2 = get_cached_model_info()
        if best_version:
            return {
                "model_name": "DeliveryTimePredictor",
                "version": best_version.version,
                "r2_score": highest_r2,
                "model_loaded": model is not None,
                "scaler_loaded": scaler is not None
            }
        else:
            return {"error": "No model info available"}
    except Exception as e:
        return {"error": f"Failed to fetch model info: {str(e)}"}

def get_health_status():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    }

def prune_model_registry(model_name: str, logger):
    client = MlflowClient()
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        version_scores = []
        for version in all_versions:
            r2 = version.tags.get("r2_score")
            try:
                r2_val = float(r2) if r2 else -float("inf")
                version_scores.append((version.version, r2_val))
            except Exception:
                logger.warning(f"Skipping version {version.version} due to invalid r2_score.")

        version_scores.sort(key=lambda x: x[1], reverse=True)
        versions_to_delete = version_scores[3:]  # keep top 3

        for version_num, r2_val in versions_to_delete:
            logger.info(f"🗑️ Deleting model version {version_num} (R2: {r2_val})")
            client.delete_model_version(name=model_name, version=str(version_num))

    except Exception as e:
        logger.error(f"❌ Failed to prune registry: {e}", exc_info=True)

def retrain_with_feedback(logger):
    try:
        params = load_params()
        X_train, y_train, X_test, y_test = load_featured_data(params, logger)

        feedback_path = "data/feedback/feedback.csv"
        if os.path.exists(feedback_path):
            feedback_df = pd.read_csv(feedback_path)
            feedback_X = feedback_df.drop(columns=['actual_delivery_time', 'predicted_delivery_time', 'prediction_id', 'timestamp', 'prediction_error'], errors='ignore')
            feedback_y = feedback_df['actual_delivery_time']
            logger.info(f"✅ Loaded {len(feedback_df)} feedback samples")
            X_train = pd.concat([X_train, feedback_X], ignore_index=True)
            y_train = pd.concat([y_train, feedback_y], ignore_index=True)
        else:
            logger.warning("⚠️ No feedback data found, using only original data")

        all_trained_models = []

        models = get_models(params)
        for name, model_ in models.items():
            model_.fit(X_train, y_train)
            r2, mse = train_and_evaluate(model_, X_train, y_train, X_test, y_test, logger)
            all_trained_models.append((f"Original_{name}", model_, r2))

        client = MlflowClient()
        registry_name = params['model_registry']['name']
        versions = client.get_latest_versions(registry_name, stages=["None"])
        for version in versions:
            try:
                loaded_model = mlflow_load_model(version.source)
                loaded_model.fit(X_train, y_train)
                r2, mse = train_and_evaluate(loaded_model, X_train, y_train, X_test, y_test, logger)
                all_trained_models.append((f"Registry_v{version.version}", loaded_model, r2))
            except Exception as e:
                logger.warning(f"⚠️ Failed to retrain model v{version.version}: {e}")

        top_models = sorted(all_trained_models, key=lambda x: x[2], reverse=True)[:3]
        results = []

        for tag, model_, r2 in top_models:
            model_info = mlflow.sklearn.log_model(sk_model=model_, artifact_path=f"top_model_{tag}")
            manage_model_registry(model_name=registry_name, model_info=model_info, challenger_r2=r2, logger=logger)
            results.append({"model_tag": tag, "r2_score": r2})

        return {
            "message": "✅ Retrained and updated top models.",
            "top_models": results
        }

    except Exception as e:
        logger.error("❌ Retraining failed", exc_info=True)
        return {"error": str(e)}



def ensure_directories():
    """Create necessary directories"""
    os.makedirs("data/results", exist_ok=True)
    os.makedirs("data/feedback", exist_ok=True)