package com.aqi_prediction.presentation.ui.theme

import androidx.compose.ui.graphics.Color

// Theme Palette
val DARK_BG = Color(0xFF0F172A)
val DARK_CARD_BG = Color(0xFF1E293B)
val DARK_CARD_STROKE = Color(0xFF334155)
val DARK_TEXT_PRIMARY = Color(0xFFF8FAFC)
val DARK_TEXT_SECONDARY = Color(0xFF94A3B8)

val LIGHT_BG = Color(0xFFF8FAFC)
val LIGHT_CARD_BG = Color(0xFFFFFFFF)
val LIGHT_CARD_STROKE = Color(0xFFE2E8F0)
val LIGHT_TEXT_PRIMARY = Color(0xFF0F172A)
val LIGHT_TEXT_SECONDARY = Color(0xFF64748B)

// Shared Accents
val ACCENT_BLUE = Color(0xFF38BDF8)
val ACCENT_GREEN = Color(0xFF10B981)

// Official EPA Standard AQI Colors (Matching AQI Guide Chart)
val AQI_GOOD = Color(0xFF98EC85)          // 0-50: Bright Light Green
val AQI_MODERATE = Color(0xFFFFC000)      // 51-100: Golden Yellow
val AQI_SENSITIVE = Color(0xFFFF7E00)     // 101-150: Vivid Orange
val AQI_UNHEALTHY = Color(0xFFE51A1A)     // 151-200: Vibrant Red
val AQI_VERY_UNHEALTHY = Color(0xFF8F3F97) // 201-300: Deep Purple
val AQI_HAZARDOUS = Color(0xFF660014)      // 301-500: Deep Maroon

// Status
val STATUS_ONLINE = Color(0xFF10B981)
val STATUS_OFFLINE = Color(0xFFEF4444)
