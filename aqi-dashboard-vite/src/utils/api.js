// API utility functions for the enhanced AQI dashboard
const API_BASE_URL = "http://127.0.0.1:5000/"

const apiCall = async (endpoint, data = null) => {
  const url = `${API_BASE_URL}${endpoint}`
  const options = {
    method: data ? "POST" : "GET",
    headers: {
      "Content-Type": "application/json",
    },
    ...(data && { body: JSON.stringify({ pollutants: data }) }),
  }

  const response = await fetch(url, options)
  if (!response.ok) {
    throw new Error(`API Error: ${response.status} - ${response.statusText}`)
  }
  return response.json()
}

export const getCurrentAQI = (pollutants) => apiCall("/predict", pollutants)
export const get24HourForecast = (pollutants) => apiCall("/24hours", pollutants)
export const get7DayForecast = (pollutants) => apiCall("/7days", pollutants)
export const healthCheck = () => apiCall("/health")
export const getApiInfo = () => apiCall("/info")
