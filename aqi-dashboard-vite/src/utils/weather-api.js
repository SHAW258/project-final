const WEATHERBIT_API_KEY = "6f3cb45f7cad4bea90af850219a0b633" // Replace with your actual API key
const WEATHERBIT_BASE_URL = "https://api.weatherbit.io/v2.0"

export const getWeatherData = async (lat, lon) => {
  const url = `${WEATHERBIT_BASE_URL}/current?lat=${lat}&lon=${lon}&key=${WEATHERBIT_API_KEY}`
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Weather API Error: ${response.status} - ${response.statusText}`)
  }
  const data = await response.json()
  return data.data[0] // Return the first (and only) data point
}

export const getAirQualityData = async (lat, lon) => {
  const url = `${WEATHERBIT_BASE_URL}/airquality?lat=${lat}&lon=${lon}&key=${WEATHERBIT_API_KEY}`
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Air Quality API Error: ${response.status} - ${response.statusText}`)
  }
  const data = await response.json()
  return data.data[0] // Return the first (and only) data point
}

export const getCurrentLocation = () => {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error("Geolocation is not supported by this browser"))
      return
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        resolve({
          lat: position.coords.latitude,
          lon: position.coords.longitude,
        })
      },
      (error) => {
        reject(error)
      },
      { timeout: 10000, enableHighAccuracy: true },
    )
  })
}

export const estimatePollutantsFromWeather = (weatherData, airQualityData) => {
  // This function now returns formatted values as strings to match input expectations
  return {
    "PM2.5": (10 + (weatherData?.temp || 0) * 0.1).toFixed(1),
    PM10: (20 + (weatherData?.wind_spd || 0) * 0.5).toFixed(1),
    NO2: (5 + (weatherData?.rh || 0) * 0.05).toFixed(1),
    SO2: (2 + (weatherData?.pres || 1000) * 0.001).toFixed(1),
    CO: (0.5 + (weatherData?.vis || 10) * 0.01).toFixed(2),
    O3: (40 + (weatherData?.temp || 0) * 0.2).toFixed(1),
  }
}

export const getWeatherIcon = (code) => {
  // Map Weatherbit weather codes to Lucide icons (or simple emojis)
  switch (code) {
    case 200: // Thunderstorm with light rain
    case 201: // Thunderstorm with rain
    case 202: // Thunderstorm with heavy rain
    case 230: // Thunderstorm with light drizzle
    case 231: // Thunderstorm with drizzle
    case 232: // Thunderstorm with heavy drizzle
      return "⛈️"
    case 300: // Light drizzle
    case 301: // Drizzle
    case 302: // Heavy Drizzle
      return "🌧️"
    case 500: // Light rain
    case 501: // Moderate rain
    case 502: // Heavy rain
    case 511: // Freezing rain
    case 520: // Light shower rain
    case 521: // Shower rain
    case 522: // Heavy shower rain
      return "☔"
    case 600: // Light snow
    case 601: // Snow
    case 602: // Heavy snow
    case 610: // Mix snow/rain
    case 611: // Sleet
    case 612: // Light sleet showers
    case 613: // Sleet showers
    case 621: // Snow showers
    case 622: // Heavy snow showers
    case 623: // Ice pellets
      return "❄️"
    case 700: // Mist
    case 711: // Smoke
    case 721: // Haze
    case 731: // Sand/dust
    case 741: // Fog
    case 751: // Sand
    case 801: // Few clouds
    case 802: // Scattered clouds
    case 803: // Broken clouds
    case 804: // Overcast clouds
      return "☁️"
    case 800: // Clear sky
    default:
      return "☀️"
  }
}