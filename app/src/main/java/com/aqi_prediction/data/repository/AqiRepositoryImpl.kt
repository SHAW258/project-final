package com.aqi_prediction.data.repository

import com.aqi_prediction.data.api.BackendApiService
import com.aqi_prediction.data.api.OpenMeteoApiService
import com.aqi_prediction.data.dto.PredictRequestDto
import com.aqi_prediction.data.mapper.toBackendPollutantsMap
import com.aqi_prediction.data.mapper.toDomain
import com.aqi_prediction.data.mapper.toPollutantItems
import com.aqi_prediction.data.mapper.toRawPollutants
import com.aqi_prediction.domain.error.AppError
import com.aqi_prediction.domain.model.CityLocation
import com.aqi_prediction.domain.model.CurrentPrediction
import com.aqi_prediction.domain.model.ForecastItem
import com.aqi_prediction.domain.model.RawPollutants
import com.aqi_prediction.domain.repository.AqiRepository
import com.aqi_prediction.domain.repository.ResultData
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.Random
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.roundToInt

/**
 * Concrete implementation of AqiRepository returning structured AppError instances.
 */
@Singleton
class AqiRepositoryImpl @Inject constructor(
    private val openMeteoService: OpenMeteoApiService,
    private val backendService: BackendApiService
) : AqiRepository {

    override val presetCities = listOf(
        CityLocation("Kolkata", "India", 22.5726, 88.3639),
        CityLocation("New Delhi", "India", 28.6139, 77.2090),
        CityLocation("Mumbai", "India", 19.0760, 72.8777),
        CityLocation("Bengaluru", "India", 12.9716, 77.5946),
        CityLocation("London", "United Kingdom", 51.5074, -0.1278),
        CityLocation("New York", "USA", 40.7128, -74.0060),
        CityLocation("Tokyo", "Japan", 35.6762, 139.6503),
        CityLocation("Sydney", "Australia", -33.8688, 151.2093),
        CityLocation("Paris", "France", 48.8566, 2.3522),
        CityLocation("Beijing", "China", 39.9042, 116.4074),
        CityLocation("Dubai", "UAE", 25.2048, 55.2708)
    )

    @Volatile
    private var lastHourlyDto: com.aqi_prediction.data.dto.OpenMeteoHourlyDto? = null

    override suspend fun getLiveAQIPrediction(lat: Double, lon: Double): ResultData {
        return try {
            // Step 1: Fetch Live Pollutants DTO from Open-Meteo API
            val liveResp = openMeteoService.getLivePollutants(
                lat = lat,
                lon = lon,
                current = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
                hourly = "pm2_5,pm10,us_aqi",
                forecastDays = 7
            )
            if (!liveResp.isSuccessful || liveResp.body()?.current == null) {
                return ResultData.Error(
                    AppError.ApiError(
                        httpCode = liveResp.code(),
                        errorMessage = "Open-Meteo API returned invalid response (HTTP ${liveResp.code()})."
                    )
                )
            }

            // Cache Open-Meteo hourly forecast for fallbacks
            lastHourlyDto = liveResp.body()?.hourly

            // Step 2: Use Mapper to convert DTO -> Domain RawPollutants
            val rawPollutants: RawPollutants = liveResp.body()!!.current!!.toRawPollutants()
            val pollutantsMap = rawPollutants.toBackendPollutantsMap()

            // Step 3: Send DTO request to Hosted Render Backend ML Model (retry while waking up)
            var predictResp: retrofit2.Response<com.aqi_prediction.data.dto.PredictApiResponseDto>? = null
            for (attempt in 1..3) {
                try {
                    val resp = backendService.predictAqi(PredictRequestDto(pollutantsMap))
                    if (resp.isSuccessful && resp.body()?.prediction != null) {
                        predictResp = resp
                        break
                    }
                } catch (e: Exception) {
                    // Render backend may be waking up from cold sleep
                }
                if (attempt < 3) {
                    kotlinx.coroutines.delay(2000)
                }
            }

            val predictionResult: CurrentPrediction
            if (predictResp != null && predictResp.isSuccessful && predictResp.body()?.prediction != null) {
                // Map DTO -> Domain CurrentPrediction
                predictionResult = predictResp.body()!!.prediction!!.toDomain()
            } else {
                // Fallback local AQI calculation if backend is waking up or offline
                predictionResult = calculateFallbackAqi(rawPollutants.pm25, rawPollutants.pm10)
            }

            val pollutantList = rawPollutants.toPollutantItems()

            ResultData.Success(
                prediction = predictionResult,
                pollutants = pollutantList,
                rawCurrent = rawPollutants,
                pollutantsMap = pollutantsMap
            )
        } catch (e: IOException) {
            ResultData.Error(AppError.NetworkError())
        } catch (e: Exception) {
            ResultData.Error(
                AppError.UnknownError(causeMsg = e.localizedMessage ?: "An unexpected error occurred.")
            )
        }
    }

    override suspend fun get24HourForecast(pollutantsMap: Map<String, Double>): List<ForecastItem> {
        return try {
            val response = backendService.get24HourForecast(PredictRequestDto(pollutantsMap))
            if (response.isSuccessful && !response.body()?.forecast.isNullOrEmpty()) {
                response.body()!!.forecast.map { it.toDomain() }
            } else {
                build24HourFromOpenMeteo() ?: generateFallbackForecast(24, pollutantsMap["PM2.5"] ?: 35.0)
            }
        } catch (e: Exception) {
            build24HourFromOpenMeteo() ?: generateFallbackForecast(24, pollutantsMap["PM2.5"] ?: 35.0)
        }
    }

    override suspend fun get7DayForecast(pollutantsMap: Map<String, Double>): List<ForecastItem> {
        return try {
            val response = backendService.get7DayForecast(PredictRequestDto(pollutantsMap))
            if (response.isSuccessful && !response.body()?.forecast.isNullOrEmpty()) {
                response.body()!!.forecast.map { it.toDomain() }
            } else {
                build7DayFromOpenMeteo() ?: generateFallbackForecast(7, pollutantsMap["PM2.5"] ?: 35.0, isDaily = true)
            }
        } catch (e: Exception) {
            build7DayFromOpenMeteo() ?: generateFallbackForecast(7, pollutantsMap["PM2.5"] ?: 35.0, isDaily = true)
        }
    }

    private fun build24HourFromOpenMeteo(): List<ForecastItem>? {
        val hourly = lastHourlyDto ?: return null
        if (hourly.time.isEmpty()) return null

        val list = mutableListOf<ForecastItem>()
        val inputFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm", Locale.US)
        val outputFormat = SimpleDateFormat("HH:00", Locale.US)

        val maxItems = minOf(24, hourly.time.size)
        for (i in 0 until maxItems) {
            val rawTime = hourly.time[i]
            val pm25Val = hourly.pm25.getOrNull(i) ?: 35.0
            val pm10Val = hourly.pm10.getOrNull(i) ?: 50.0
            val usAqiVal = hourly.usAqi.getOrNull(i) ?: calculateFallbackAqi(pm25Val, pm10Val).aqi

            val formattedTime = try {
                val date = inputFormat.parse(rawTime)
                if (date != null) outputFormat.format(date) else rawTime
            } catch (e: Exception) {
                rawTime
            }

            val (level, color, emoji) = when {
                usAqiVal <= 50 -> Triple("Good", "#00E400", "🟢")
                usAqiVal <= 100 -> Triple("Moderate", "#FFFF00", "🟡")
                usAqiVal <= 150 -> Triple("Unhealthy (Sensitive)", "#FF7E00", "🟠")
                usAqiVal <= 200 -> Triple("Unhealthy", "#FF0000", "🔴")
                usAqiVal <= 300 -> Triple("Very Unhealthy", "#8F3F97", "🟣")
                else -> Triple("Hazardous", "#7E0023", "🤎")
            }

            list.add(
                ForecastItem(
                    time = formattedTime,
                    aqi = usAqiVal,
                    level = level,
                    color = color,
                    emoji = emoji
                )
            )
        }
        return if (list.isNotEmpty()) list else null
    }

    private fun build7DayFromOpenMeteo(): List<ForecastItem>? {
        val hourly = lastHourlyDto ?: return null
        if (hourly.time.isEmpty()) return null

        val list = mutableListOf<ForecastItem>()
        val inputFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm", Locale.US)
        val dayOutputFormat = SimpleDateFormat("EEE, MMM d", Locale.US)

        val chunks = hourly.time.indices.chunked(24).take(7)
        for (chunk in chunks) {
            if (chunk.isEmpty()) continue
            val firstIndex = chunk.first()
            val rawTime = hourly.time[firstIndex]

            var maxAqi = 0
            for (idx in chunk) {
                val pm25Val = hourly.pm25.getOrNull(idx) ?: 35.0
                val pm10Val = hourly.pm10.getOrNull(idx) ?: 50.0
                val aqi = hourly.usAqi.getOrNull(idx) ?: calculateFallbackAqi(pm25Val, pm10Val).aqi
                if (aqi > maxAqi) maxAqi = aqi
            }
            if (maxAqi == 0) maxAqi = 50

            val formattedDay = try {
                val date = inputFormat.parse(rawTime)
                if (date != null) dayOutputFormat.format(date) else rawTime
            } catch (e: Exception) {
                rawTime
            }

            val (level, color, emoji) = when {
                maxAqi <= 50 -> Triple("Good", "#00E400", "🟢")
                maxAqi <= 100 -> Triple("Moderate", "#FFFF00", "🟡")
                maxAqi <= 150 -> Triple("Unhealthy (Sensitive)", "#FF7E00", "🟠")
                maxAqi <= 200 -> Triple("Unhealthy", "#FF0000", "🔴")
                maxAqi <= 300 -> Triple("Very Unhealthy", "#8F3F97", "🟣")
                else -> Triple("Hazardous", "#7E0023", "🤎")
            }

            list.add(
                ForecastItem(
                    time = formattedDay,
                    aqi = maxAqi,
                    level = level,
                    color = color,
                    emoji = emoji
                )
            )
        }
        return if (list.isNotEmpty()) list else null
    }

    override suspend fun isBackendHealthy(): Boolean {
        return try {
            val resp = backendService.checkHealth()
            resp.isSuccessful
        } catch (e: Exception) {
            false
        }
    }

    private fun calculateFallbackAqi(pm25: Double, pm10: Double): CurrentPrediction {
        val aqiPm25 = calculateSubIndex(pm25, floatArrayOf(0f, 12f, 35.4f, 55.4f, 150.4f, 250.4f, 500f), intArrayOf(0, 50, 100, 150, 200, 300, 500))
        val aqiPm10 = calculateSubIndex(pm10, floatArrayOf(0f, 54f, 154f, 254f, 354f, 424f, 604f), intArrayOf(0, 50, 100, 150, 200, 300, 500))
        val aqi = maxOf(aqiPm25, aqiPm10)

        val (level, color, emoji, healthMsg) = when {
            aqi <= 50 -> Quadruple("Good", "#00E400", "🟢", "Air quality is satisfactory, and air pollution poses little or no risk.")
            aqi <= 100 -> Quadruple("Moderate", "#FFFF00", "🟡", "Air quality is acceptable. However, sensitive individuals may experience minor symptoms.")
            aqi <= 150 -> Quadruple("Unhealthy for Sensitive Groups", "#FF7E00", "🟠", "Members of sensitive groups may experience health effects. General public is less likely to be affected.")
            aqi <= 200 -> Quadruple("Unhealthy", "#FF0000", "🔴", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.")
            aqi <= 300 -> Quadruple("Very Unhealthy", "#8F3F97", "🟣", "Health alert: The risk of health effects is increased for everyone.")
            else -> Quadruple("Hazardous", "#7E0023", "🤎", "Health warning of emergency conditions: everyone is more likely to be affected.")
        }

        val df = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
        return CurrentPrediction(
            aqi = aqi,
            timestamp = df.format(Date()),
            level = level,
            color = color,
            emoji = emoji,
            healthMessage = healthMsg
        )
    }

    private fun calculateSubIndex(c: Double, breakpoints: FloatArray, aqiValues: IntArray): Int {
        for (i in 0 until breakpoints.size - 1) {
            if (c >= breakpoints[i] && c <= breakpoints[i + 1]) {
                val cLow = breakpoints[i]
                val cHigh = breakpoints[i + 1]
                val aLow = aqiValues[i]
                val aHigh = aqiValues[i + 1]
                return (((aHigh - aLow) / (cHigh - cLow)) * (c - cLow) + aLow).roundToInt()
            }
        }
        return 500
    }

    private fun generateFallbackForecast(count: Int, currentPm25: Double, isDaily: Boolean = false): List<ForecastItem> {
        val list = mutableListOf<ForecastItem>()
        val cal = Calendar.getInstance()
        val sdf = if (isDaily) SimpleDateFormat("EEE, MMM d", Locale.US) else SimpleDateFormat("HH:00", Locale.US)

        val baseAqi = (currentPm25 * 3.0).roundToInt().coerceIn(20, 350)
        val random = Random()

        for (i in 1..count) {
            if (isDaily) cal.add(Calendar.DAY_OF_YEAR, 1) else cal.add(Calendar.HOUR_OF_DAY, 1)
            val variation = random.nextInt(30) - 15
            val fAqi = (baseAqi + variation).coerceIn(15, 450)

            val (level, color, emoji) = when {
                fAqi <= 50 -> Triple("Good", "#00E400", "🟢")
                fAqi <= 100 -> Triple("Moderate", "#FFFF00", "🟡")
                fAqi <= 150 -> Triple("Unhealthy (Sensitive)", "#FF7E00", "🟠")
                fAqi <= 200 -> Triple("Unhealthy", "#FF0000", "🔴")
                else -> Triple("Very Unhealthy", "#8F3F97", "🟣")
            }

            list.add(
                ForecastItem(
                    time = sdf.format(cal.time),
                    aqi = fAqi,
                    level = level,
                    color = color,
                    emoji = emoji
                )
            )
        }
        return list
    }
}

private data class Quadruple<A, B, C, D>(val first: A, val second: B, val third: C, val fourth: D)
