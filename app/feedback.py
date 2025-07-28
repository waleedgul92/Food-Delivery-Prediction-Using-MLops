import os
import logging
from datetime import datetime
import pandas as pd
from app.schema import FeedbackInput
import threading
from app.model_handler import setup_logging
logger = setup_logging()
def get_feedback_stats():
    """Get statistics about received feedback"""
    try:
        feedback_file = "data/feedback/feedback.csv"
        if not os.path.exists(feedback_file):
            return {"message": "No feedback data available"}
        
        df = pd.read_csv(feedback_file)
        
        stats = {
            "total_feedback_count": len(df),
            "average_prediction_error": round(df['prediction_error'].mean(), 2),
            "min_prediction_error": round(df['prediction_error'].min(), 2),
            "max_prediction_error": round(df['prediction_error'].max(), 2),
            "latest_feedback": df.iloc[-1]['timestamp'] if len(df) > 0 else None
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        return {"error": f"Error getting feedback stats: {str(e)}"}

def save_feedback_to_csv(feedback_data: dict):
    """Save feedback data to CSV file with thread safety"""
    try:
        # Add timestamp if not provided
        if 'timestamp' not in feedback_data or not feedback_data['timestamp']:
            feedback_data['timestamp'] = datetime.now().isoformat()
        
        feedback_file = "data/feedback/feedback.csv"
        df = pd.DataFrame([feedback_data])
        
        # Thread-safe file writing
        with threading.Lock():
            if not os.path.exists(feedback_file):
                df.to_csv(feedback_file, index=False)
                logger.info(f"📁 Created new feedback file: {feedback_file}")
            else:
                df.to_csv(feedback_file, mode='a', header=False, index=False)
        
        logger.info(f"💾 Feedback saved to {feedback_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving feedback: {e}", exc_info=True)
        return False