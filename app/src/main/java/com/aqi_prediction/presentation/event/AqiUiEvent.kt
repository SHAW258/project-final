package com.aqi_prediction.presentation.event

import com.aqi_prediction.domain.error.AppError

/**
 * Single-shot UI Events in presentation/event package.
 */
sealed class AqiUiEvent {
    data class ShowToast(val message: String) : AqiUiEvent()
    data class ShowErrorDialog(val error: AppError) : AqiUiEvent()
    object DataRefreshed : AqiUiEvent()
}
