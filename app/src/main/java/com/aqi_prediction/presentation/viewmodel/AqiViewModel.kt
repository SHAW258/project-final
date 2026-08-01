package com.aqi_prediction.presentation.viewmodel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aqi_prediction.domain.error.AppError
import com.aqi_prediction.domain.model.CityLocation
import com.aqi_prediction.domain.model.CurrentPrediction
import com.aqi_prediction.domain.model.ForecastItem
import com.aqi_prediction.domain.model.PollutantItem
import com.aqi_prediction.domain.repository.AqiRepository
import com.aqi_prediction.domain.repository.ResultData
import com.aqi_prediction.presentation.event.AqiUiEvent
import com.aqi_prediction.presentation.state.AqiUiState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class AqiViewModel @Inject constructor(
    private val repository: AqiRepository
) : ViewModel() {

    val presetCities = repository.presetCities

    private val _uiState = MutableLiveData<AqiUiState>(AqiUiState.Loading())
    val uiState: LiveData<AqiUiState> = _uiState

    private val _eventFlow = MutableSharedFlow<AqiUiEvent>()
    val eventFlow: SharedFlow<AqiUiEvent> = _eventFlow

    private val _forecastList = MutableLiveData<List<ForecastItem>>(emptyList())
    val forecastList: LiveData<List<ForecastItem>> = _forecastList

    private val _isForecastLoading = MutableLiveData<Boolean>(false)
    val isForecastLoading: LiveData<Boolean> = _isForecastLoading

    private val _isBackendOnline = MutableLiveData<Boolean>(true)
    val isBackendOnline: LiveData<Boolean> = _isBackendOnline

    private val _selectedCity = MutableLiveData<CityLocation>(presetCities[0]) // Default Kolkata
    val selectedCity: LiveData<CityLocation> = _selectedCity

    private val _currentTab = MutableLiveData<Int>(0) // 0: 24h, 1: 7d
    val currentTab: LiveData<Int> = _currentTab

    init {
        checkBackendHealth()
        loadDataForCity(presetCities[0])
    }

    fun checkBackendHealth() {
        viewModelScope.launch {
            val isHealthy = repository.isBackendHealthy()
            _isBackendOnline.postValue(isHealthy)
        }
    }

    fun loadDataForCity(city: CityLocation) {
        _selectedCity.value = city
        fetchAqiData(city.latitude, city.longitude, city.name)
    }

    fun loadDataForCoordinates(lat: Double, lon: Double, label: String = "GPS Location") {
        fetchAqiData(lat, lon, label)
    }

    fun refreshCurrentLocation() {
        val city = _selectedCity.value ?: presetCities[0]
        fetchAqiData(city.latitude, city.longitude, city.name)
    }

    fun setForecastTab(tabIndex: Int, pollutantsMap: Map<String, Double>?) {
        _currentTab.value = tabIndex
        if (pollutantsMap == null) return

        viewModelScope.launch {
            _isForecastLoading.postValue(true)
            val list = if (tabIndex == 0) {
                repository.get24HourForecast(pollutantsMap)
            } else {
                repository.get7DayForecast(pollutantsMap)
            }
            _forecastList.postValue(list)
            _isForecastLoading.postValue(false)
        }
    }

    private fun fetchAqiData(lat: Double, lon: Double, locationName: String) {
        viewModelScope.launch {
            // Step 1: Open-Meteo Pollutant Feed
            _uiState.value = AqiUiState.Loading(
                step = 1,
                message = "Connecting to Open-Meteo Live Air Quality Sensors…"
            )
            
            // Step 2: Render ML Model Prediction / Cold Start Wakeup
            _uiState.value = AqiUiState.Loading(
                step = 2,
                message = "Waking Up & Connecting to Hosted XGBoost ML Model on Render…"
            )

            val result = repository.getLiveAQIPrediction(lat, lon)

            when (result) {
                is ResultData.Success -> {
                    // Step 3: Processing Forecast Engine
                    _uiState.value = AqiUiState.Loading(
                        step = 3,
                        message = "Generating 24-Hour & 7-Day ML Forecasts…"
                    )

                    _isForecastLoading.postValue(true)
                    val activeTab = _currentTab.value ?: 0
                    val forecast = if (activeTab == 1) {
                        repository.get7DayForecast(result.pollutantsMap)
                    } else {
                        repository.get24HourForecast(result.pollutantsMap)
                    }
                    _forecastList.postValue(forecast)
                    _isForecastLoading.postValue(false)

                    // Re-check backend online indicator
                    checkBackendHealth()

                    _uiState.postValue(
                        AqiUiState.Success(
                            prediction = result.prediction,
                            pollutants = result.pollutants,
                            locationName = locationName,
                            latitude = lat,
                            longitude = lon,
                            pollutantsMap = result.pollutantsMap
                        )
                    )
                    _eventFlow.emit(AqiUiEvent.DataRefreshed)
                }
                is ResultData.Error -> {
                    _uiState.postValue(AqiUiState.Error(result.error))
                    _eventFlow.emit(AqiUiEvent.ShowErrorDialog(result.error))
                }
            }
        }
    }
}
