package com.aqi_prediction.data.dto

import com.google.gson.annotations.SerializedName

/**
 * Data Transfer Objects (DTOs) for Render ML Backend API.
 */
data class PredictRequestDto(
    val pollutants: Map<String, Double>
)

data class CurrentPredictionResponseDto(
    val aqi: Int = 0,
    val timestamp: String = "",
    val level: String = "Good",
    val color: String = "#00E400",
    val emoji: String = "🟢",
    @SerializedName("health_message") val healthMessage: String = ""
)

data class PredictApiResponseDto(
    val prediction: CurrentPredictionResponseDto? = null,
    val plot: String? = "",
    val status: String = "success",
    val message: String? = null
)

data class HealthResponseDto(
    val status: String = "online"
)

data class ForecastItemDto(
    val time: String = "",
    val aqi: Int = 0,
    val level: String = "",
    val color: String = "#00E400",
    val emoji: String = "🟢"
)

data class ForecastApiResponseDto(
    val forecast: List<ForecastItemDto> = emptyList(),
    val status: String = "success"
)


