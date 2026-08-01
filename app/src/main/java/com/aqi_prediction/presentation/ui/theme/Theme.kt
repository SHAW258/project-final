package com.aqi_prediction.presentation.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val DarkColorScheme = darkColorScheme(
    primary = AccentBlue,
    onPrimary = DarkTextPrimary,
    secondary = AccentGreen,
    onSecondary = DarkBg,
    background = DarkBg,
    onBackground = DarkTextPrimary,
    surface = DarkCardBg,
    onSurface = DarkTextPrimary,
    surfaceVariant = DarkCardBg,
    onSurfaceVariant = DarkTextSecondary,
    outline = DarkCardStroke
)

private val LightColorScheme = lightColorScheme(
    primary = AccentBlue,
    onPrimary = LightTextPrimary,
    secondary = AccentGreen,
    onSecondary = LightBg,
    background = LightBg,
    onBackground = LightTextPrimary,
    surface = LightCardBg,
    onSurface = LightTextPrimary,
    surfaceVariant = LightCardBg,
    onSurfaceVariant = LightTextSecondary,
    outline = LightCardStroke
)

@Composable
fun AQIVisionTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}

