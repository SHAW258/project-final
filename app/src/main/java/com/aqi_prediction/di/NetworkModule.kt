package com.aqi_prediction.di

import com.aqi_prediction.core.network.ApiConstants
import com.aqi_prediction.core.network.NetworkClientProvider
import com.aqi_prediction.core.network.RetrofitBuilder
import com.aqi_prediction.data.api.BackendApiService
import com.aqi_prediction.data.api.OpenMeteoApiService
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        return NetworkClientProvider.createOkHttpClient()
    }

    @Provides
    @Singleton
    fun provideOpenMeteoApiService(okHttpClient: OkHttpClient): OpenMeteoApiService {
        return RetrofitBuilder.createService(
            baseUrl = ApiConstants.OPEN_METEO_BASE_URL,
            okHttpClient = okHttpClient,
            serviceClass = OpenMeteoApiService::class.java
        )
    }

    @Provides
    @Singleton
    fun provideBackendApiService(okHttpClient: OkHttpClient): BackendApiService {
        return RetrofitBuilder.createService(
            baseUrl = ApiConstants.RENDER_BACKEND_BASE_URL,
            okHttpClient = okHttpClient,
            serviceClass = BackendApiService::class.java
        )
    }
}


