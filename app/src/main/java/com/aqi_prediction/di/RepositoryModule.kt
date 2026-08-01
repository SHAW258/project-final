package com.aqi_prediction.di

import com.aqi_prediction.data.repository.AqiRepositoryImpl
import com.aqi_prediction.domain.repository.AqiRepository
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt module binding AqiRepository interface to AqiRepositoryImpl implementation.
 */
@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    @Binds
    @Singleton
    abstract fun bindAqiRepository(
        impl: AqiRepositoryImpl
    ): AqiRepository
}

