from pydantic import BaseModel, Field
from typing import Dict, List, Any

class PollutantInput(BaseModel):
    PM2_5: float = Field(..., alias="PM2.5", description="PM2.5 concentration in ug/m3")
    PM10: float = Field(..., description="PM10 concentration in ug/m3")
    NO2: float = Field(..., description="NO2 concentration in ug/m3")
    SO2: float = Field(..., description="SO2 concentration in ug/m3")
    CO: float = Field(..., description="CO concentration in mg/m3")
    O3: float = Field(..., description="O3 concentration in ug/m3")

    class Config:
        populate_by_name = True

class PredictRequest(BaseModel):
    pollutants: Dict[str, float] = Field(..., description="Dictionary containing PM2.5, PM10, NO2, SO2, CO, O3")

class AQIDetails(BaseModel):
    level: str
    color: str
    emoji: str
    health_message: str

class CurrentPredictionResponse(BaseModel):
    aqi: int
    timestamp: str
    level: str
    color: str
    emoji: str
    health_message: str

class PredictApiResponse(BaseModel):
    prediction: CurrentPredictionResponse
    plot: str
    status: str

class Forecast24HourItem(BaseModel):
    timestamp: str
    predicted_aqi: int
    hour_ahead: int
    level: str
    color: str
    emoji: str
    health_message: str

class Forecast24HourResponse(BaseModel):
    forecasts: List[Forecast24HourItem]
    statistics: Dict[str, Any]
    recommendations: Dict[str, Any]
    plot: str | None = ""

class Forecast7DayItem(BaseModel):
    date: str
    predicted_aqi: int
    level: str
    color: str
    emoji: str
    health_message: str
    is_weekend: bool
    day_name: str

class Forecast7DayResponse(BaseModel):
    forecasts: List[Forecast7DayItem]
    weekly_statistics: Dict[str, Any]
    plot: str | None = ""

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    database_status: str
    server: str

class ApiInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    endpoints: List[str]
    model_status: str
    database: str
