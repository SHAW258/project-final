'use client';

import { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import {
  Loader2,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Heart,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  Info,
  Wind,
  Sun,
} from 'lucide-react';
import WeatherIntegration from './weather-integration';

const pollutantInfo = {
  'PM2.5': {
    unit: 'μg/m³',
    description: 'Fine Particulate Matter',
    icon: '🌫️',
    healthyRange: '0-12',
    dangerLevel: 35,
  },
  PM10: {
    unit: 'μg/m³',
    description: 'Coarse Particulate Matter',
    icon: '💨',
    healthyRange: '0-54',
    dangerLevel: 154,
  },
  NO2: {
    unit: 'ppb',
    description: 'Nitrogen Dioxide',
    icon: '🏭',
    healthyRange: '0-53',
    dangerLevel: 100,
  },
  SO2: {
    unit: 'ppb',
    description: 'Sulfur Dioxide',
    icon: '⚗️',
    healthyRange: '0-35',
    dangerLevel: 75,
  },
  CO: {
    unit: 'ppm',
    description: 'Carbon Monoxide',
    icon: '🚗',
    healthyRange: '0-4.4',
    dangerLevel: 9.4,
  },
  O3: {
    unit: 'ppm',
    description: 'Ozone',
    icon: '☀️',
    healthyRange: '0-54',
    dangerLevel: 70,
  },
};

// Enhanced API functions
const API_BASE_URL = 'http://127.0.0.1:5000';

const apiCall = async (endpoint, data = null) => {
  const url = `${API_BASE_URL}${endpoint}`;
  const options = {
    method: data ? 'POST' : 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    ...(data && { body: JSON.stringify({ pollutants: data }) }),
  };

  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`API Error: ${response.status} - ${response.statusText}`);
  }
  return response.json();
};

const getCurrentAQI = pollutants => apiCall('/predict', pollutants);
const get24HourForecast = pollutants => apiCall('/24hours', pollutants);
const get7DayForecast = pollutants => apiCall('/7days', pollutants);
const healthCheck = () => apiCall('/health');
const getApiInfo = () => apiCall('/info');

// Weatherbit API configuration
const WEATHERBIT_API_KEY = 'YOUR_WEATHERBIT_API_KEY'; // Replace with your actual API key
const WEATHERBIT_BASE_URL = 'https://api.weatherbit.io/v2.0';

const getCurrentLocation = () => {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation is not supported by this browser'));
      return;
    }

    navigator.geolocation.getCurrentPosition(
      position => {
        resolve({
          lat: position.coords.latitude,
          lon: position.coords.longitude,
        });
      },
      error => {
        reject(error);
      },
      { timeout: 10000, enableHighAccuracy: true },
    );
  });
};

export function EnhancedAQIDashboard() {
  const [pollutants, setPollutants] = useState({
    'PM2.5': '',
    PM10: '',
    NO2: '',
    SO2: '',
    CO: '',
    O3: '',
  });
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [forecasts24h, setForecasts24h] = useState([]);
  const [forecasts7d, setForecasts7d] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [weeklyStats, setWeeklyStats] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [plotImage, setPlotImage] = useState('');
  const [loading, setLoading] = useState({});
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');
  const [apiInfo, setApiInfo] = useState(null);
  const [location, setLocation] = useState({ lat: null, lon: null, city: '' });
  const [weatherLoading, setWeatherLoading] = useState(false);
  // Check if Flask server is running when component mounts
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const [healthData, infoData] = await Promise.all([
          healthCheck(),
          getApiInfo(),
        ]);
        setApiStatus('connected');
        setApiInfo(infoData);
        console.log('✅ Successfully connected to Flask backend');
        console.log('📊 API Info:', infoData);
      } catch (err) {
        setApiStatus('disconnected');
        setError(
          'Unable to connect to Flask server. Please ensure the realistic-aqi-api.py is running on http://localhost:5000',
        );
        console.error('❌ Failed to connect to Flask backend:', err);
      }
    };

    checkApiHealth();
  }, []);

  const handlePollutantChange = (pollutant, value) => {
    setPollutants(prev => ({
      ...prev,
      [pollutant]: value,
    }));
  };

  const validateInputs = () => {
    // Check if any field is empty
    const hasEmptyFields = Object.values(pollutants).some(
      value => value === '',
    );

    if (hasEmptyFields) {
      setError('Please fill in all pollutant fields before predicting.');
      return false;
    }

    // Convert all values to numbers for API call
    const numericPollutants = {};
    for (const [key, value] of Object.entries(pollutants)) {
      const numValue = Number.parseFloat(value);
      if (isNaN(numValue) || numValue < 0) {
        setError(`Invalid value for ${key}. Please enter a positive number.`);
        return false;
      }
      numericPollutants[key] = numValue;
    }

    return numericPollutants;
  };

  const predictCurrent = async () => {
    setLoading(prev => ({ ...prev, current: true }));
    setError('');

    const validatedData = validateInputs();
    if (!validatedData) {
      setLoading(prev => ({ ...prev, current: false }));
      return;
    }

    try {
      console.log('🔮 Requesting current AQI prediction...');
      console.log('📊 Pollutant data being sent:', validatedData);

      const data = await getCurrentAQI(validatedData);
      console.log('🎉 Received prediction:', data);

      if (data.prediction) {
        setCurrentPrediction(data.prediction);
        setPlotImage(data.plot || '');
      }
    } catch (err) {
      console.error('❌ Prediction failed:', err);
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, current: false }));
    }
  };

  const predict24Hours = async () => {
    setLoading(prev => ({ ...prev, '24h': true }));
    setError('');

    const validatedData = validateInputs();
    if (!validatedData) {
      setLoading(prev => ({ ...prev, '24h': false }));
      return;
    }

    try {
      console.log('📅 Requesting 24-hour forecast...');
      const data = await get24HourForecast(validatedData);
      console.log('📈 Received 24h forecast:', data);

      if (data.forecasts) {
        setForecasts24h(data.forecasts);
        setStatistics(data.statistics);
        setRecommendations(data.recommendations);
        setPlotImage(data.plot || '');
      }
    } catch (err) {
      console.error('❌ 24h forecast failed:', err);
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, '24h': false }));
    }
  };

  const predict7Days = async () => {
    setLoading(prev => ({ ...prev, '7d': true }));
    setError('');

    const validatedData = validateInputs();
    if (!validatedData) {
      setLoading(prev => ({ ...prev, '7d': false }));
      return;
    }

    try {
      console.log('📅 Requesting 7-day forecast...');
      const data = await get7DayForecast(validatedData);
      console.log('📈 Received 7d forecast:', data);

      if (data.forecasts) {
        setForecasts7d(data.forecasts);
        setWeeklyStats(data.weekly_statistics);
        setPlotImage(data.plot || '');
      }
    } catch (err) {
      console.error('❌ 7d forecast failed:', err);
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, '7d': false }));
    }
  };

  const getAQIStatusIcon = level => {
    switch (level) {
      case 'Good':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'Moderate':
        return <Activity className="w-5 h-5 text-yellow-500" />;
      case 'Unhealthy for Sensitive Groups':
        return <AlertTriangle className="w-5 h-5 text-orange-500" />;
      case 'Unhealthy':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'Very Unhealthy':
        return <XCircle className="w-5 h-5 text-purple-500" />;
      case 'Hazardous':
        return <XCircle className="w-5 h-5 text-red-800" />;
      default:
        return <Activity className="w-5 h-5" />;
    }
  };

  const getTrendIcon = trend => {
    switch (trend?.toLowerCase()) {
      case 'improving':
        return <TrendingDown className="w-4 h-4 text-green-500" />;
      case 'worsening':
        return <TrendingUp className="w-4 h-4 text-red-500" />;
      default:
        return <Minus className="w-4 h-4 text-gray-500" />;
    }
  };

  const getPollutantStatus = (value, pollutant) => {
    const info = pollutantInfo[pollutant];
    if (value <= info.dangerLevel * 0.3)
      return { status: 'good', color: 'text-green-600' };
    if (value <= info.dangerLevel * 0.7)
      return { status: 'moderate', color: 'text-yellow-600' };
    if (value <= info.dangerLevel)
      return { status: 'unhealthy', color: 'text-orange-600' };
    return { status: 'dangerous', color: 'text-red-600' };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4">
      {/* Enhanced Header */}
      <div className="text-center mb-6">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">
          🌍 Enhanced AQI Dashboard
        </h1>
        <p className="text-gray-600">
          Real-time Air Quality Index Prediction & Forecasting
        </p>
        {apiInfo && (
          <Badge variant="outline" className="mt-2">
            {apiInfo.name} v{apiInfo.version}
          </Badge>
        )}
      </div>

      {/* API Status Indicator */}
      <div className="flex items-center justify-center gap-2 mb-4">
        <div
          className={`w-3 h-3 rounded-full ${
            apiStatus === 'connected'
              ? 'bg-green-500 animate-pulse'
              : apiStatus === 'disconnected'
              ? 'bg-red-500'
              : 'bg-yellow-500'
          }`}
        />
        <span className="text-sm text-gray-600">
          Backend Status:{' '}
          {apiStatus === 'connected'
            ? 'Connected ✅'
            : apiStatus === 'disconnected'
            ? 'Disconnected ❌'
            : 'Checking...'}
        </span>
      </div>

      {/* Weather Integration */}
      <WeatherIntegration onPollutantsUpdate={() => {}} />

      {/* Error Alert */}
      {error && (
        <Alert className="border-red-200 bg-red-50 mb-4">
          <AlertTriangle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-800">{error}</AlertDescription>
        </Alert>
      )}

      {/* Enhanced Pollutant Input Section */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wind className="w-5 h-5" />
            Pollutant Concentrations
          </CardTitle>
          <CardDescription>
            Enter current pollutant levels to generate AQI predictions and
            forecasts
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(pollutants).map(([pollutant, value]) => {
              const info = pollutantInfo[pollutant];
              const numValue = Number.parseFloat(value) || 0;
              const status = getPollutantStatus(numValue, pollutant);

              return (
                <div key={pollutant} className="space-y-3">
                  <Label
                    htmlFor={pollutant}
                    className="flex items-center gap-2"
                  >
                    <span className="text-lg">{info.icon}</span>
                    <span className="font-semibold">{pollutant}</span>
                    <Badge variant="secondary" className="text-xs">
                      {info.unit}
                    </Badge>
                  </Label>
                  <Input
                    id={pollutant}
                    type="number"
                    step="0.1"
                    min="0"
                    onChange={e =>
                      handlePollutantChange(pollutant, e.target.value)
                    }
                    className="w-full"
                    placeholder={`Enter ${pollutant} level`}
                  />
                  <div className="space-y-1">
                    <p className="text-xs text-gray-500">{info.description}</p>
                    <p className="text-xs text-gray-400">
                      Healthy range: {info.healthyRange} {info.unit}
                    </p>
                    {value && numValue > 0 && (
                      <p className={`text-xs font-medium ${status.color}`}>
                        Status:{' '}
                        {status.status.charAt(0).toUpperCase() +
                          status.status.slice(1)}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <Separator />

          <div className="flex gap-2 mt-4">
            <Button
              variant="outline"
              onClick={() =>
                setPollutants({
                  'PM2.5': '',
                  PM10: '',
                  NO2: '',
                  SO2: '',
                  CO: '',
                  O3: '',
                })
              }
              className="flex-1"
            >
              Clear All
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Prediction Tabs */}
      <Tabs defaultValue="current" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="current" className="flex items-center gap-2">
            <Activity className="w-4 h-4" />
            Current AQI
          </TabsTrigger>
          <TabsTrigger value="24hours" className="flex items-center gap-2">
            <Clock className="w-4 h-4" />
            24-Hour Forecast
          </TabsTrigger>
          <TabsTrigger value="7days" className="flex items-center gap-2">
            <Sun className="w-4 h-4" />
            7-Day Forecast
          </TabsTrigger>
        </TabsList>

        {/* Enhanced Current Prediction Tab */}
        <TabsContent value="current" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Current AQI Prediction
              </CardTitle>
              <CardDescription>
                Get instant AQI prediction with health recommendations
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={predictCurrent}
                disabled={loading.current || apiStatus !== 'connected'}
                className="w-full"
                size="lg"
              >
                {loading.current ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing Air Quality...
                  </>
                ) : (
                  <>
                    <Activity className="mr-2 h-4 w-4" />
                    Predict Current AQI
                  </>
                )}
              </Button>

              {currentPrediction && (
                <div className="space-y-6">
                  {/* Main AQI Display */}
                  <Card
                    className="border-2"
                    style={{ borderColor: currentPrediction.color }}
                  >
                    <CardContent className="pt-6">
                      <div className="text-center space-y-4">
                        <div
                          className="text-8xl font-bold"
                          style={{ color: currentPrediction.color }}
                        >
                          {currentPrediction.aqi}
                        </div>
                        <div className="flex items-center justify-center gap-3">
                          {getAQIStatusIcon(currentPrediction.level)}
                          <Badge
                            className="text-white text-xl px-6 py-3"
                            style={{ backgroundColor: currentPrediction.color }}
                          >
                            {currentPrediction.level}
                          </Badge>
                        </div>
                        {currentPrediction.health_message && (
                          <Alert className="mt-4">
                            <Heart className="h-4 w-4" />
                            <AlertDescription className="text-left">
                              <strong>Health Advisory:</strong>{' '}
                              {currentPrediction.health_message}
                            </AlertDescription>
                          </Alert>
                        )}
                        <p className="text-sm text-gray-600">
                          Predicted at{' '}
                          {new Date(
                            currentPrediction.timestamp,
                          ).toLocaleString()}
                        </p>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Visualization */}
                  {plotImage && (
                    <Card>
                      <CardHeader>
                        <CardTitle>
                          AQI Analysis & Pollutant Breakdown
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <img
                          src={`data:image/png;base64,${plotImage}`}
                          alt="AQI Visualization"
                          className="w-full h-auto rounded-lg shadow-lg"
                        />
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Enhanced 24-Hour Forecast Tab */}
        <TabsContent value="24hours" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5" />
                24-Hour AQI Forecast
              </CardTitle>
              <CardDescription>
                Hourly predictions with peak analysis and recommendations
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={predict24Hours}
                disabled={loading['24h'] || apiStatus !== 'connected'}
                className="w-full"
                size="lg"
              >
                {loading['24h'] ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating 24-Hour Forecast...
                  </>
                ) : (
                  <>
                    <Clock className="mr-2 h-4 w-4" />
                    Generate 24-Hour Forecast
                  </>
                )}
              </Button>

              {/* Enhanced Statistics */}
              {statistics && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <Card>
                    <CardContent className="pt-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">
                          {Math.round(statistics.average)}
                        </div>
                        <p className="text-sm text-gray-600">Average</p>
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                          {statistics.min}
                        </div>
                        <p className="text-sm text-gray-600">Minimum</p>
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">
                          {statistics.max}
                        </div>
                        <p className="text-sm text-gray-600">Maximum</p>
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-600">
                          {statistics.median}
                        </div>
                        <p className="text-sm text-gray-600">Median</p>
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-orange-600">
                          {Math.round(statistics.std)}
                        </div>
                        <p className="text-sm text-gray-600">Std Dev</p>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}

              {/* Recommendations */}
              {recommendations && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Info className="w-5 h-5" />
                      Smart Recommendations
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="flex items-center gap-2 p-3 bg-green-50 rounded-lg">
                        <CheckCircle className="w-5 h-5 text-green-600" />
                        <div>
                          <p className="font-medium text-green-800">
                            Best Time for Outdoor Activities
                          </p>
                          <p className="text-sm text-green-600">
                            {recommendations.best_time_outdoor}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 p-3 bg-red-50 rounded-lg">
                        <XCircle className="w-5 h-5 text-red-600" />
                        <div>
                          <p className="font-medium text-red-800">
                            Avoid Outdoor Activities
                          </p>
                          <p className="text-sm text-red-600">
                            {recommendations.avoid_time}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 p-3 bg-blue-50 rounded-lg">
                        {getTrendIcon(recommendations.overall_trend)}
                        <div>
                          <p className="font-medium text-blue-800">
                            Overall Trend
                          </p>
                          <p className="text-sm text-blue-600">
                            {recommendations.overall_trend}
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Enhanced Visualization */}
              {plotImage && (
                <Card>
                  <CardHeader>
                    <CardTitle>24-Hour Analysis Dashboard</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <img
                      src={`data:image/png;base64,${plotImage}`}
                      alt="24-Hour Forecast Analysis"
                      className="w-full h-auto rounded-lg shadow-lg"
                    />
                  </CardContent>
                </Card>
              )}

              {/* Hourly Breakdown */}
              {forecasts24h.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Hourly Breakdown (Next 12 Hours)</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto">
                      {forecasts24h.slice(0, 12).map((forecast, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-4 rounded-lg border-2 hover:shadow-md transition-shadow"
                          style={{
                            borderLeftColor: forecast.color,
                            borderLeftWidth: '4px',
                          }}
                        >
                          <div className="flex items-center gap-3">
                            <div className="text-2xl">{forecast.emoji}</div>
                            <div>
                              <p className="font-medium">
                                {new Date(
                                  forecast.timestamp,
                                ).toLocaleTimeString([], {
                                  hour: '2-digit',
                                  minute: '2-digit',
                                })}
                              </p>
                              <p className="text-sm text-gray-600">
                                {forecast.level}
                              </p>
                              {forecast.health_message && (
                                <p
                                  className="text-xs text-gray-500 mt-1 max-w-32 truncate"
                                  title={forecast.health_message}
                                >
                                  {forecast.health_message}
                                </p>
                              )}
                            </div>
                          </div>
                          <div className="text-right">
                            <p
                              className="text-xl font-bold"
                              style={{ color: forecast.color }}
                            >
                              {forecast.predicted_aqi}
                            </p>
                            <Badge variant="outline" className="text-xs mt-1">
                              +{forecast.hour_ahead}h
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Enhanced 7-Day Forecast Tab */}
        <TabsContent value="7days" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sun className="w-5 h-5" />
                7-Day AQI Forecast
              </CardTitle>
              <CardDescription>
                Weekly air quality predictions with trend analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={predict7Days}
                disabled={loading['7d'] || apiStatus !== 'connected'}
                className="w-full"
                size="lg"
              >
                {loading['7d'] ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating 7-Day Forecast...
                  </>
                ) : (
                  <>
                    <Sun className="mr-2 h-4 w-4" />
                    Generate 7-Day Forecast
                  </>
                )}
              </Button>

              {/* Weekly Statistics */}
              {weeklyStats && (
                <Card>
                  <CardHeader>
                    <CardTitle>Weekly Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-3 bg-blue-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">
                          {weeklyStats.average}
                        </div>
                        <p className="text-sm text-blue-800">Weekly Average</p>
                      </div>
                      <div className="text-center p-3 bg-green-50 rounded-lg">
                        <div className="text-lg font-bold text-green-600">
                          {weeklyStats.best_day}
                        </div>
                        <p className="text-sm text-green-800">Best Day</p>
                      </div>
                      <div className="text-center p-3 bg-red-50 rounded-lg">
                        <div className="text-lg font-bold text-red-600">
                          {weeklyStats.worst_day}
                        </div>
                        <p className="text-sm text-red-800">Worst Day</p>
                      </div>
                      <div className="text-center p-3 bg-purple-50 rounded-lg">
                        <div className="flex items-center justify-center gap-1">
                          {getTrendIcon(weeklyStats.trend)}
                          <span className="text-lg font-bold text-purple-600">
                            {weeklyStats.trend}
                          </span>
                        </div>
                        <p className="text-sm text-purple-800">Trend</p>
                      </div>
                    </div>

                    {weeklyStats.weekday_avg && weeklyStats.weekend_avg && (
                      <div className="mt-4 grid grid-cols-2 gap-4">
                        <div className="text-center p-3 bg-gray-50 rounded-lg">
                          <div className="text-xl font-bold text-gray-600">
                            {weeklyStats.weekday_avg}
                          </div>
                          <p className="text-sm text-gray-800">
                            Weekday Average
                          </p>
                        </div>
                        <div className="text-center p-3 bg-indigo-50 rounded-lg">
                          <div className="text-xl font-bold text-indigo-600">
                            {weeklyStats.weekend_avg}
                          </div>
                          <p className="text-sm text-indigo-800">
                            Weekend Average
                          </p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Enhanced Visualization */}
              {plotImage && (
                <Card>
                  <CardHeader>
                    <CardTitle>7-Day Forecast Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <img
                      src={`data:image/png;base64,${plotImage}`}
                      alt="7-Day Forecast Analysis"
                      className="w-full h-auto rounded-lg shadow-lg"
                    />
                  </CardContent>
                </Card>
              )}

              {/* Enhanced Daily Breakdown */}
              {forecasts7d.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Daily Breakdown</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {forecasts7d.map((forecast, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-4 rounded-lg border-2 hover:shadow-md transition-all"
                          style={{ borderColor: forecast.color }}
                        >
                          <div className="flex items-center gap-4">
                            <div className="text-3xl">{forecast.emoji}</div>
                            <div>
                              <p className="font-semibold text-lg">
                                {new Date(forecast.date).toLocaleDateString(
                                  'en-US',
                                  {
                                    weekday: 'long',
                                    month: 'short',
                                    day: 'numeric',
                                  },
                                )}
                              </p>
                              <p className="text-sm text-gray-600">
                                {forecast.level}
                              </p>
                              {forecast.health_message && (
                                <p className="text-xs text-gray-500 mt-1 max-w-64">
                                  {forecast.health_message}
                                </p>
                              )}
                              {forecast.is_weekend && (
                                <Badge variant="secondary" className="mt-1">
                                  Weekend
                                </Badge>
                              )}
                            </div>
                          </div>
                          <div className="text-right">
                            <p
                              className="text-4xl font-bold"
                              style={{ color: forecast.color }}
                            >
                              {forecast.predicted_aqi}
                            </p>
                            <Badge
                              className="text-white mt-2"
                              style={{ backgroundColor: forecast.color }}
                            >
                              AQI Level
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
