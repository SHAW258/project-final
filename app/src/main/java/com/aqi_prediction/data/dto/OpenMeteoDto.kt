package com.aqi_prediction.data.dto

import com.google.gson.annotations.SerializedName

/**
 * Data Transfer Objects (DTOs) for Open-Meteo Air Quality API.
 */
data class OpenMeteoResponseDto(
    val latitude: Double? = null,
    val longitude: Double? = null,
    val current: OpenMeteoCurrentDto? = null,
    val hourly: OpenMeteoHourlyDto? = null,
    @SerializedName("current_units") val currentUnits: Map<String, String>? = null
)

data class OpenMeteoCurrentDto(
    val time: String? = null,
    @SerializedName("pm2_5") val pm25: Double = 0.0,
    @SerializedName("pm10") val pm10: Double = 0.0,
    @SerializedName("nitrogen_dioxide") val no2: Double = 0.0,
    @SerializedName("sulphur_dioxide") val so2: Double = 0.0,
    @SerializedName("carbon_monoxide") val co: Double = 0.0,
    @SerializedName("ozone") val o3: Double = 0.0
)

data class OpenMeteoHourlyDto(
    val time: List<String> = emptyList(),
    @SerializedName("pm2_5") val pm25: List<Double?> = emptyList(),
    @SerializedName("pm10") val pm10: List<Double?> = emptyList(),
    @SerializedName("us_aqi") val usAqi: List<Int?> = emptyList()
)


