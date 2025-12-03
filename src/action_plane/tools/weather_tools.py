"""Weather-related tools - Safe risk level examples.

These tools demonstrate safe tool implementations that can be
auto-approved without user confirmation.
"""

import os
import requests
from datetime import datetime
from typing import Dict, Any

from pydantic import BaseModel, Field

from ...core.domain.tools import RiskLevel
from ..tool_registry import governor_tool


def _wind_direction(degrees: float) -> str:
    """Convert wind direction in degrees to compass direction."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]


class WeatherSchema(BaseModel):
    """Schema for weather tool arguments."""
    location: str = Field(..., description="City and state/country for weather lookup")


class ForecastSchema(BaseModel):
    """Schema for forecast tool arguments."""
    location: str = Field(..., description="City and state/country for forecast lookup")
    days: int = Field(default=3, ge=1, le=7, description="Number of days for forecast (1-7)")


@governor_tool(
    name="get_weather",
    description="Get current weather conditions for a specified location",
    risk_level=RiskLevel.SAFE,
    args_schema=WeatherSchema,
    category="data",
    tags=["weather", "information", "safe"],
    timeout_seconds=10.0,
    max_retries=2
)
def get_weather(location: str) -> dict:
    """Get current weather for a location.
    
    This is a safe tool that provides weather information without
    any security risks or side effects.
    
    Args:
        location: City and state/country to get weather for
        
    Returns:
        Dictionary with current weather information
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "error": "OpenWeatherMap API key not configured",
            "message": "Set OPENWEATHERMAP_API_KEY environment variable"
        }
    
    try:
        # Call OpenWeatherMap current weather API
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric"  # Use metric for consistency
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to Fahrenheit for US users while keeping Celsius
        temp_c = data["main"]["temp"]
        temp_f = (temp_c * 9/5) + 32
        
        weather_data = {
            "location": f"{data['name']}, {data['sys']['country']}",
            "temperature": f"{temp_f:.0f}°F ({temp_c:.0f}°C)",
            "condition": data["weather"][0]["description"].title(),
            "humidity": f"{data['main']['humidity']}%",
            "wind": f"{data['wind']['speed']} m/s {_wind_direction(data['wind'].get('deg', 0))}",
            "pressure": f"{data['main']['pressure']} hPa",
            "visibility": f"{data.get('visibility', 'N/A')/1000:.1f} km" if 'visibility' in data else "N/A",
            "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        }
        
        return {
            "status": "success",
            "data": weather_data,
            "source": "openweathermap_api"
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error": "API request failed",
            "message": str(e)
        }
    except (KeyError, ValueError) as e:
        return {
            "status": "error", 
            "error": "Invalid API response",
            "message": str(e)
        }


@governor_tool(
    name="get_forecast",
    description="Get weather forecast for multiple days",
    risk_level=RiskLevel.SAFE,
    args_schema=ForecastSchema,
    category="data",
    tags=["weather", "forecast", "information", "safe"],
    timeout_seconds=15.0,
    max_retries=2
)
def get_forecast(location: str, days: int = 3) -> dict:
    """Get weather forecast for specified number of days.
    
    Args:
        location: City and state/country to get forecast for
        days: Number of days for forecast (1-7)
        
    Returns:
        Dictionary with forecast information
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "error": "OpenWeatherMap API key not configured",
            "message": "Set OPENWEATHERMAP_API_KEY environment variable"
        }
    
    # Limit days to maximum supported by API (5 days)
    days = min(days, 5)
    
    try:
        # Call OpenWeatherMap forecast API
        url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric"
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse forecast data (API returns 3-hour intervals for 5 days)
        forecast_days = []
        current_date = None
        daily_data = {}
        
        for item in data["list"]:
            # Parse date from timestamp
            date_str = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
            
            if date_str != current_date:
                # Save previous day if exists
                if current_date and daily_data:
                    forecast_days.append(daily_data)
                    if len(forecast_days) >= days:
                        break
                
                # Start new day
                current_date = date_str
                temp_c = item["main"]["temp"]
                temp_f = (temp_c * 9/5) + 32
                
                daily_data = {
                    "date": date_str,
                    "day": datetime.fromtimestamp(item["dt"]).strftime("%A"),
                    "condition": item["weather"][0]["description"].title(),
                    "high": f"{temp_f:.0f}°F ({temp_c:.0f}°C)",
                    "low": f"{temp_f:.0f}°F ({temp_c:.0f}°C)",  # Will be updated
                    "humidity": f"{item['main']['humidity']}%",
                    "precipitation": f"{item['pop']*100:.0f}%"
                }
            else:
                # Update min/max temps for the day
                if daily_data:
                    temp_c = item["main"]["temp"]
                    temp_f = (temp_c * 9/5) + 32
                    
                    current_high = float(daily_data["high"].split("°F")[0])
                    current_low = float(daily_data["low"].split("°F")[0])
                    
                    if temp_f > current_high:
                        daily_data["high"] = f"{temp_f:.0f}°F ({temp_c:.0f}°C)"
                    if temp_f < current_low:
                        daily_data["low"] = f"{temp_f:.0f}°F ({temp_c:.0f}°C)"
        
        # Add last day if not already added
        if daily_data and len(forecast_days) < days:
            forecast_days.append(daily_data)
        
        return {
            "status": "success",
            "location": f"{data['city']['name']}, {data['city']['country']}",
            "forecast_days": len(forecast_days),
            "data": forecast_days,
            "source": "openweathermap_api"
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error": "API request failed",
            "message": str(e)
        }
    except (KeyError, ValueError) as e:
        return {
            "status": "error",
            "error": "Invalid API response", 
            "message": str(e)
        }