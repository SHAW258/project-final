package com.aqi_prediction.data.mapper

import com.aqi_prediction.data.dto.CurrentPredictionResponseDto
import com.aqi_prediction.data.dto.ForecastItemDto
import com.aqi_prediction.data.dto.OpenMeteoCurrentDto
import com.aqi_prediction.domain.model.CurrentPrediction
import com.aqi_prediction.domain.model.ForecastItem
import com.aqi_prediction.domain.model.PollutantItem
import com.aqi_prediction.domain.model.RawPollutants
import java.util.Locale

/**
 * Mapper extension functions converting DTO payloads into Domain models.
 */
fun CurrentPredictionResponseDto.toDomain(): CurrentPrediction {
    return CurrentPrediction(
        aqi = aqi,
        timestamp = timestamp,
        level = level,
        color = color,
        emoji = emoji,
        healthMessage = healthMessage
    )
}

fun ForecastItemDto.toDomain(): ForecastItem {
    return ForecastItem(
        time = time,
        aqi = aqi,
        level = level,
        color = color,
        emoji = emoji
    )
}

fun OpenMeteoCurrentDto.toRawPollutants(): RawPollutants {
    return RawPollutants(
        pm25 = pm25,
        pm10 = pm10,
        no2 = no2,
        so2 = so2,
        coMgM3 = co / 1000.0,
        o3 = o3
    )
}

fun RawPollutants.toBackendPollutantsMap(): Map<String, Double> {
    return mapOf(
        "PM2.5" to pm25,
        "PM10" to pm10,
        "NO2" to no2,
        "SO2" to so2,
        "CO" to coMgM3,
        "O3" to o3
    )
}

fun RawPollutants.toPollutantItems(): List<PollutantItem> {
    return listOf(
        PollutantItem(
            key = "PM2.5",
            name = "Fine Particles (PM2.5)",
            rawValue = pm25,
            formattedValue = String.format(Locale.US, "%.1f", pm25),
            unit = "µg/m³",
            category = getCategory("PM2.5", pm25),
            statusColorHex = getColor("PM2.5", pm25)
        ),
        PollutantItem(
            key = "PM10",
            name = "Coarse Particles (PM10)",
            rawValue = pm10,
            formattedValue = String.format(Locale.US, "%.1f", pm10),
            unit = "µg/m³",
            category = getCategory("PM10", pm10),
            statusColorHex = getColor("PM10", pm10)
        ),
        PollutantItem(
            key = "NO2",
            name = "Nitrogen Dioxide (NO₂)",
            rawValue = no2,
            formattedValue = String.format(Locale.US, "%.1f", no2),
            unit = "µg/m³",
            category = getCategory("NO2", no2),
            statusColorHex = getColor("NO2", no2)
        ),
        PollutantItem(
            key = "SO2",
            name = "Sulfur Dioxide (SO₂)",
            rawValue = so2,
            formattedValue = String.format(Locale.US, "%.1f", so2),
            unit = "µg/m³",
            category = getCategory("SO2", so2),
            statusColorHex = getColor("SO2", so2)
        ),
        PollutantItem(
            key = "CO",
            name = "Carbon Monoxide (CO)",
            rawValue = coMgM3,
            formattedValue = String.format(Locale.US, "%.3f", coMgM3),
            unit = "mg/m³",
            category = getCategory("CO", coMgM3),
            statusColorHex = getColor("CO", coMgM3)
        ),
        PollutantItem(
            key = "O3",
            name = "Ozone (O₃)",
            rawValue = o3,
            formattedValue = String.format(Locale.US, "%.1f", o3),
            unit = "µg/m³",
            category = getCategory("O3", o3),
            statusColorHex = getColor("O3", o3)
        )
    )
}

private fun getCategory(key: String, valDouble: Double): String {
    return when (key) {
        "PM2.5" -> when {
            valDouble <= 12.0 -> "Good"
            valDouble <= 35.4 -> "Moderate"
            valDouble <= 55.4 -> "Unhealthy to Sensitive Groups"
            valDouble <= 150.4 -> "Unhealthy"
            else -> "High Risk"
        }
        "PM10" -> when {
            valDouble <= 54.0 -> "Good"
            valDouble <= 154.0 -> "Moderate"
            valDouble <= 254.0 -> "Unhealthy to Sensitive Groups"
            else -> "High Risk"
        }
        else -> if (valDouble < 50.0) "Good" else "Moderate"
    }
}

private fun getColor(key: String, valDouble: Double): String {
    return when (key) {
        "PM2.5" -> when {
            valDouble <= 12.0 -> "#98EC85"
            valDouble <= 35.4 -> "#FFC000"
            valDouble <= 55.4 -> "#FF7E00"
            valDouble <= 150.4 -> "#E51A1A"
            else -> "#8F3F97"
        }
        "PM10" -> when {
            valDouble <= 54.0 -> "#98EC85"
            valDouble <= 154.0 -> "#FFC000"
            valDouble <= 254.0 -> "#FF7E00"
            else -> "#E51A1A"
        }
        else -> "#98EC85"
    }
}


