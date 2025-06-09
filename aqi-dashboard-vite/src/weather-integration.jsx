"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Cloud,
  Thermometer,
  Droplets,
  Eye,
  WindIcon,
  Activity,
  MapPin,
  RefreshCw,
  Loader2,
  AlertTriangle,
} from "lucide-react"
import {
  getWeatherData,
  getAirQualityData,
  getCurrentLocation,
  estimatePollutantsFromWeather,
  getWeatherIcon,
} from "@/utils/weather-api"

export default function WeatherIntegration({ onPollutantsUpdate }) {
  const [weatherData, setWeatherData] = useState(null)
  const [airQualityData, setAirQualityData] = useState(null)
  const [location, setLocation] = useState({ lat: 0, lon: 0, city: "" })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const fetchWeatherAndAirQuality = async (lat, lon) => {
    setLoading(true)
    setError("")

    try {
      let coords = { lat: lat || location.lat, lon: lon || location.lon }

      // If no coordinates provided, get current location
      if (!coords.lat || !coords.lon) {
        try {
          coords = await getCurrentLocation()
        } catch (locationError) {
          console.warn("Could not get location:", locationError)
          // Default to New York City
          coords = { lat: 40.7128, lon: -74.006 }
        }
      }

      // Fetch weather and air quality data in parallel
      const [weather, airQuality] = await Promise.allSettled([
        getWeatherData(coords.lat, coords.lon),
        getAirQualityData(coords.lat, coords.lon),
      ])

      if (weather.status === "fulfilled") {
        setWeatherData(weather.value)
        setLocation({
          lat: coords.lat,
          lon: coords.lon,
          city: `${weather.value.city_name}, ${weather.value.country_code}`,
        })

        // Update pollutants based on weather/air quality data
        // const airQualityValue = airQuality.status === "fulfilled" ? airQuality.value : null
        // const estimatedPollutants = estimatePollutantsFromWeather(weather.value, airQualityValue)
        // onPollutantsUpdate(estimatedPollutants)

        if (airQuality.status === "fulfilled") {
          setAirQualityData(airQuality.value)
        }
      } else {
        throw new Error("Failed to fetch weather data")
      }
    } catch (err) {
      setError(`Weather data error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Auto-fetch weather data on component mount
  useEffect(() => {
    fetchWeatherAndAirQuality()
  }, [])

  if (!weatherData && !loading) {
    return (
      <Card className="mb-6 border-blue-200 bg-blue-50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-blue-800">
            <Cloud className="w-5 h-5" />
            Weather Integration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center space-y-3">
            <p className="text-sm text-blue-700">Get real-time weather data to enhance your air quality predictions</p>
            <Button onClick={() => fetchWeatherAndAirQuality()} className="w-full">
              <Cloud className="mr-2 h-4 w-4" />
              Load Weather Data
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (loading) {
    return (
      <Card className="mb-6 border-blue-200 bg-blue-50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-blue-800">
            <Cloud className="w-5 h-5" />
            Weather Integration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center gap-2 py-4">
            <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
            <span className="text-blue-700">Fetching real-time weather data...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Alert className="mb-6 border-orange-200 bg-orange-50">
        <AlertTriangle className="h-4 w-4 text-orange-600" />
        <AlertDescription className="text-orange-800">
          {error}
          <Button variant="outline" size="sm" onClick={() => fetchWeatherAndAirQuality()} className="ml-2">
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <Card className="mb-6 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-600 text-white">
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-white">
          <div className="flex items-center gap-2">
            {getWeatherIcon(weatherData?.weather?.code || 800)}
            <span>Real-time Weather Conditions</span>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="flex items-center gap-1">
              <MapPin className="w-3 h-3" />
              {location.city}
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => fetchWeatherAndAirQuality()}
              disabled={loading}
              className="text-white hover:bg-white/20"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="flex items-center gap-2 bg-white/10 rounded-lg p-3 backdrop-blur-sm">
            <Thermometer className="w-5 h-5" />
            <div>
              <p className="text-sm opacity-80">Temperature</p>
              <p className="text-xl font-bold">{weatherData?.temp}°C</p>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-white/10 rounded-lg p-3 backdrop-blur-sm">
            <Droplets className="w-5 h-5" />
            <div>
              <p className="text-sm opacity-80">Humidity</p>
              <p className="text-xl font-bold">{weatherData?.rh}%</p>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-white/10 rounded-lg p-3 backdrop-blur-sm">
            <WindIcon className="w-5 h-5" />
            <div>
              <p className="text-sm opacity-80">Wind Speed</p>
              <p className="text-xl font-bold">{weatherData?.wind_spd} m/s</p>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-white/10 rounded-lg p-3 backdrop-blur-sm">
            <Eye className="w-5 h-5" />
            <div>
              <p className="text-sm opacity-80">Visibility</p>
              <p className="text-xl font-bold">{weatherData?.vis} km</p>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-white/10 rounded-lg p-3 backdrop-blur-sm">
            <Activity className="w-5 h-5" />
            <div>
              <p className="text-sm opacity-80">Pressure</p>
              <p className="text-xl font-bold">{weatherData?.pres} mb</p>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-white/10 rounded-lg p-3 backdrop-blur-sm">
            <Cloud className="w-5 h-5" />
            <div>
              <p className="text-sm opacity-80">Conditions</p>
              <p className="text-sm font-medium">{weatherData?.weather?.description}</p>
            </div>
          </div>
        </div>

        {airQualityData && (
          <div className="mt-4 p-3 bg-white/10 rounded-lg backdrop-blur-sm">
            <p className="text-sm opacity-80 mb-2">Real-time Air Quality Data Available</p>
            <div className="grid grid-cols-3 md:grid-cols-6 gap-2 text-xs">
              <div>PM2.5: {airQualityData.pm25}</div>
              <div>PM10: {airQualityData.pm10}</div>
              <div>NO2: {airQualityData.no2}</div>
              <div>SO2: {airQualityData.so2}</div>
              <div>CO: {airQualityData.co}</div>
              <div>O3: {airQualityData.o3}</div>
            </div>
          </div>
        )}

        

        
      </CardContent>
    </Card>
  )
}