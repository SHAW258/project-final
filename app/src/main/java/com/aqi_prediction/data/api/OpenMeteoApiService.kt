package com.aqi_prediction.data.api

import com.aqi_prediction.data.dto.OpenMeteoResponseDto
import retrofit2.Response
import retrofit2.http.GET
import retrofit2.http.Query

interface OpenMeteoApiService {
    @GET("v1/air-quality")
    suspend fun getLivePollutants(
        @Query("latitude") lat: Double,
        @Query("longitude") lon: Double,
        @Query("current") current: String = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        @Query("hourly") hourly: String = "pm2_5,pm10,us_aqi",
        @Query("forecast_days") forecastDays: Int = 7
    ): Response<OpenMeteoResponseDto>
}


