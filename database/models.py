from sqlalchemy import Column, Integer, Float, String, Text, DateTime
from datetime import datetime
from database.database import Base

class PredictionRecord(Base):
    """Database model for storing AQI prediction requests and results in MySQL."""
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    forecast_type = Column(String(50), default="current") # 'current', '24hours', '7days'
    
    # Pollutant inputs
    pm25 = Column(Float, nullable=True)
    pm10 = Column(Float, nullable=True)
    no2 = Column(Float, nullable=True)
    so2 = Column(Float, nullable=True)
    co = Column(Float, nullable=True)
    o3 = Column(Float, nullable=True)
    
    # Prediction output
    predicted_aqi = Column(Integer, nullable=False)
    level = Column(String(100), nullable=True)
    color = Column(String(20), nullable=True)
    emoji = Column(String(20), nullable=True)
    health_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "forecast_type": self.forecast_type,
            "pollutants": {
                "PM2.5": self.pm25,
                "PM10": self.pm10,
                "NO2": self.no2,
                "SO2": self.so2,
                "CO": self.co,
                "O3": self.o3
            },
            "predicted_aqi": self.predicted_aqi,
            "level": self.level,
            "color": self.color,
            "emoji": self.emoji,
            "health_message": self.health_message,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
