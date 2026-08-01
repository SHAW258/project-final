package com.aqi_prediction.data.api

import com.aqi_prediction.data.dto.ForecastApiResponseDto
import com.aqi_prediction.data.dto.HealthResponseDto
import com.aqi_prediction.data.dto.PredictApiResponseDto
import com.aqi_prediction.data.dto.PredictRequestDto
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

interface BackendApiService {
    @POST("predict")
    suspend fun predictAqi(
        @Body request: PredictRequestDto
    ): Response<PredictApiResponseDto>

    @POST("24hours")
    suspend fun get24HourForecast(
        @Body request: PredictRequestDto
    ): Response<ForecastApiResponseDto>

    @POST("7days")
    suspend fun get7DayForecast(
        @Body request: PredictRequestDto
    ): Response<ForecastApiResponseDto>

    @GET("health")
    suspend fun checkHealth(): Response<HealthResponseDto>
}


