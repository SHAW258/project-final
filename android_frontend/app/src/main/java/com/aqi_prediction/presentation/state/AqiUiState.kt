package com.aqi_prediction.presentation.state

import com.aqi_prediction.domain.error.AppError
import com.aqi_prediction.domain.model.CurrentPrediction
import com.aqi_prediction.domain.model.PollutantItem

/**
 * UI State in presentation/state package.
 */
sealed class AqiUiState {
    data class Loading(
        val step: Int = 1,
        val message: String = "Connecting to Open-Meteo Live Air Quality Sensors…"
    ) : AqiUiState()

    data class Success(
        val prediction: CurrentPrediction,
        val pollutants: List<PollutantItem>,
        val locationName: String,
        val latitude: Double,
        val longitude: Double,
        val pollutantsMap: Map<String, Double>
    ) : AqiUiState()

    data class Error(val error: AppError) : AqiUiState()
}
