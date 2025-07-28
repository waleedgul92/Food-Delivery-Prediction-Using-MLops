from pydantic import BaseModel
from typing import Optional

class PredictionInput(BaseModel):
    Distance: float
    Preparation_Time: float
    Courier_Experience: float
    Weather_Foggy: int
    Weather_Rainy: int
    Weather_Snowy: int
    Weather_Windy: int
    Traffic_Level_Low: int
    Traffic_Level_Medium: int
    Time_of_Day_Morning: int
    Time_of_Day_Evening: int
    Time_of_Day_Night: int
    Vehicle_Type_Car: int
    Vehicle_Type_Scooter: int

class FeedbackInput(PredictionInput):
    predicted_delivery_time: float
    actual_delivery_time: float
    prediction_id: Optional[str] = None
    timestamp: Optional[str] = None