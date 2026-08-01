package com.aqi_prediction.domain.repository

import com.aqi_prediction.domain.error.AppError
import com.aqi_prediction.domain.model.CityLocation
import com.aqi_prediction.domain.model.CurrentPrediction
import com.aqi_prediction.domain.model.ForecastItem
import com.aqi_prediction.domain.model.PollutantItem
import com.aqi_prediction.domain.model.RawPollutants

/**
 * Domain Repository contract interface.
 */
interface AqiRepository {
    val presetCities: List<CityLocation>
    suspend fun getLiveAQIPrediction(lat: Double, lon: Double): ResultData
    suspend fun get24HourForecast(pollutantsMap: Map<String, Double>): List<ForecastItem>
    suspend fun get7DayForecast(pollutantsMap: Map<String, Double>): List<ForecastItem>
    suspend fun isBackendHealthy(): Boolean
}

sealed class ResultData {
    data class Success(
        val prediction: CurrentPrediction,
        val pollutants: List<PollutantItem>,
        val rawCurrent: RawPollutants,
        val pollutantsMap: Map<String, Double>
    ) : ResultData()

    data class Error(val error: AppError) : ResultData()
}
