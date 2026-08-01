package com.aqi_prediction.domain.model

/**
 * Domain Models representing business entities used by Repository, ViewModel, and Jetpack Compose UI.
 */
data class CurrentPrediction(
    val aqi: Int = 0,
    val timestamp: String = "",
    val level: String = "Good",
    val color: String = "#00E400",
    val emoji: String = "🟢",
    val healthMessage: String = ""
)

data class ForecastItem(
    val time: String = "",
    val aqi: Int = 0,
    val level: String = "",
    val color: String = "#00E400",
    val emoji: String = "🟢"
)

data class CityLocation(
    val name: String,
    val country: String,
    val latitude: Double,
    val longitude: Double
)

data class PollutantItem(
    val key: String,
    val name: String,
    val rawValue: Double,
    val formattedValue: String,
    val unit: String,
    val category: String,
    val statusColorHex: String
)

data class RawPollutants(
    val pm25: Double = 0.0,
    val pm10: Double = 0.0,
    val no2: Double = 0.0,
    val so2: Double = 0.0,
    val coMgM3: Double = 0.0,
    val o3: Double = 0.0
)


