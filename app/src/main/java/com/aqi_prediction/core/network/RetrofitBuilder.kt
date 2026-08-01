package com.aqi_prediction.core.network

import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

/**
 * Dedicated Retrofit Builder constructing Retrofit service clients.
 */
object RetrofitBuilder {

    fun <T> createService(
        baseUrl: String,
        okHttpClient: OkHttpClient,
        serviceClass: Class<T>
    ): T {
        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(serviceClass)
    }
}


