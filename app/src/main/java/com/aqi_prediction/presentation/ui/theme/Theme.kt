package com.aqi_prediction.presentation.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = ACCENT_BLUE,
    onPrimary = Color.White,
    secondary = ACCENT_GREEN,
    onSecondary = DARK_BG,
    background = DARK_BG,
    onBackground = DARK_TEXT_PRIMARY,
    surface = DARK_CARD_BG,
    onSurface = DARK_TEXT_PRIMARY,
    surfaceVariant = DARK_CARD_BG,
    onSurfaceVariant = DARK_TEXT_SECONDARY,
    outline = DARK_CARD_STROKE
)

private val LightColorScheme = lightColorScheme(
    primary = ACCENT_BLUE,
    onPrimary = Color.White,
    secondary = ACCENT_GREEN,
    onSecondary = LIGHT_BG,
    background = LIGHT_BG,
    onBackground = LIGHT_TEXT_PRIMARY,
    surface = LIGHT_CARD_BG,
    onSurface = LIGHT_TEXT_PRIMARY,
    surfaceVariant = LIGHT_CARD_BG,
    onSurfaceVariant = LIGHT_TEXT_SECONDARY,
    outline = LIGHT_CARD_STROKE
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
