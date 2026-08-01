package com.aqi_prediction.presentation.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aqi_prediction.domain.model.CityLocation
import com.aqi_prediction.domain.model.CurrentPrediction
import com.aqi_prediction.domain.model.ForecastItem
import com.aqi_prediction.domain.model.PollutantItem
import com.aqi_prediction.domain.error.AppError
import com.aqi_prediction.presentation.event.AqiUiEvent
import com.aqi_prediction.presentation.state.AqiUiState
import com.aqi_prediction.presentation.viewmodel.AqiViewModel
import com.aqi_prediction.presentation.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AQIScreen(
    viewModel: AqiViewModel,
    onGpsClick: () -> Unit
) {
    val uiState by viewModel.uiState.observeAsState(AqiUiState.Loading())
    val isBackendOnline by viewModel.isBackendOnline.observeAsState(true)
    val selectedCity by viewModel.selectedCity.observeAsState(viewModel.presetCities[0])
    val forecastList by viewModel.forecastList.observeAsState(emptyList())
    val currentTab by viewModel.currentTab.observeAsState(0)
    val isForecastLoading by viewModel.isForecastLoading.observeAsState(false)

    var showCustomDialog by remember { mutableStateOf(false) }

    Scaffold(
        containerColor = BgDark,
        floatingActionButton = {
            FloatingActionButton(
                onClick = { viewModel.refreshCurrentLocation() },
                containerColor = AccentBlue,
                contentColor = Color.White,
                shape = CircleShape
            ) {
                Text("🔄", fontSize = 20.sp)
            }
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
        ) {
            // Header Bar
            HeaderSection(
                isOnline = isBackendOnline,
                onCheckHealth = { viewModel.checkBackendHealth() }
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Location Bar
            LocationSelectorBar(
                presetCities = viewModel.presetCities,
                selectedCity = selectedCity,
                onCitySelected = { viewModel.loadDataForCity(it) },
                onGpsClick = onGpsClick,
                onCustomCoordsClick = { showCustomDialog = true }
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Content Body State
            when (val state = uiState) {
                is AqiUiState.Loading -> {
                    ThreeStepLoadingScreen(loadingState = state)
                }
                is AqiUiState.Error -> {
                    val err = state.error
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Card(
                            modifier = Modifier
                                .fillMaxWidth(0.9f)
                                .border(1.dp, AqiUnhealthy, RoundedCornerShape(20.dp)),
                            colors = CardDefaults.cardColors(containerColor = CardBg),
                            shape = RoundedCornerShape(20.dp)
                        ) {
                            Column(
                                modifier = Modifier.padding(24.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Text(
                                    text = "⚠️ ERROR [${err.errorCode}]",
                                    color = AqiUnhealthy,
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Bold,
                                    letterSpacing = 1.sp
                                )
                                Spacer(modifier = Modifier.height(12.dp))
                                Text(
                                    text = err.errorMessage,
                                    color = TextPrimary,
                                    fontSize = 14.sp,
                                    textAlign = TextAlign.Center,
                                    lineHeight = 20.sp
                                )
                                Spacer(modifier = Modifier.height(20.dp))
                                Button(
                                    onClick = { viewModel.refreshCurrentLocation() },
                                    colors = ButtonDefaults.buttonColors(containerColor = AccentBlue),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Text("🔄 Retry Connection", color = Color.White, fontWeight = FontWeight.SemiBold)
                                }
                            }
                        }
                    }
                }
                is AqiUiState.Success -> {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .verticalScroll(rememberScrollState())
                            .padding(bottom = 80.dp)
                    ) {
                        // Hero AQI Card
                        HeroAqiCard(data = state)

                        Spacer(modifier = Modifier.height(16.dp))

                        // Health Advisory Card
                        HealthAdvisoryCard(message = state.prediction.healthMessage)

                        Spacer(modifier = Modifier.height(20.dp))

                        // Live Pollutants Section
                        Text(
                            text = "Live Pollutants Breakdown",
                            color = TextPrimary,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(modifier = Modifier.height(10.dp))

                        PollutantGrid(pollutants = state.pollutants)

                        Spacer(modifier = Modifier.height(20.dp))

                        // Forecast Section
                        ForecastSection(
                            currentTab = currentTab,
                            forecastList = forecastList,
                            isLoading = isForecastLoading,
                            onTabSelected = { tab ->
                                viewModel.setForecastTab(tab, state.pollutantsMap)
                            }
                        )
                    }
                }
            }
        }
    }

    if (showCustomDialog) {
        CustomLocationDialog(
            onDismiss = { showCustomDialog = false },
            onConfirm = { lat, lon ->
                showCustomDialog = false
                val label = String.format("Coords (%.2f, %.2f)", lat, lon)
                viewModel.loadDataForCoordinates(lat, lon, label)
            }
        )
    }
}

@Composable
fun HeaderSection(isOnline: Boolean, onCheckHealth: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "🍃",
            fontSize = 28.sp
        )
        Spacer(modifier = Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = "AQI Vision",
                color = TextPrimary,
                fontSize = 20.sp,
                fontWeight = FontWeight.Black
            )
            Text(
                text = "Real-Time Air Quality & ML Forecast",
                color = TextSecondary,
                fontSize = 11.sp
            )
        }
        // Online status pill
        Surface(
            onClick = onCheckHealth,
            color = Color(0x1A38BDF8),
            shape = CircleShape,
            border = androidx.compose.foundation.BorderStroke(1.dp, Color(0x4038BDF8))
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .background(if (isOnline) StatusOnline else StatusOffline, CircleShape)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = if (isOnline) "Backend Online" else "Backend Standby",
                    color = TextPrimary,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
fun LocationSelectorBar(
    presetCities: List<CityLocation>,
    selectedCity: CityLocation,
    onCitySelected: (CityLocation) -> Unit,
    onGpsClick: () -> Unit,
    onCustomCoordsClick: () -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Dropdown Menu Container
        Surface(
            modifier = Modifier
                .weight(1f)
                .height(44.dp)
                .clickable { expanded = true },
            color = CardBg,
            shape = RoundedCornerShape(12.dp),
            border = androidx.compose.foundation.BorderStroke(1.dp, CardStroke)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "${selectedCity.name}, ${selectedCity.country}",
                    color = TextPrimary,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(" 🔻", fontSize = 12.sp, color = TextSecondary)
            }

            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false },
                modifier = Modifier.background(CardBg)
            ) {
                presetCities.forEach { city ->
                    DropdownMenuItem(
                        text = { Text("${city.name}, ${city.country}", color = TextPrimary) },
                        onClick = {
                            expanded = false
                            onCitySelected(city)
                        }
                    )
                }
            }
        }

        Spacer(modifier = Modifier.width(8.dp))

        // GPS Button
        IconButton(
            onClick = onGpsClick,
            modifier = Modifier
                .size(44.dp)
                .background(CardBg, RoundedCornerShape(12.dp))
                .border(1.dp, CardStroke, RoundedCornerShape(12.dp))
        ) {
            Text(
                text = "📍",
                fontSize = 20.sp
            )
        }

        Spacer(modifier = Modifier.width(8.dp))

        // Custom Location Button
        IconButton(
            onClick = onCustomCoordsClick,
            modifier = Modifier
                .size(44.dp)
                .background(CardBg, RoundedCornerShape(12.dp))
                .border(1.dp, CardStroke, RoundedCornerShape(12.dp))
        ) {
            Text("📌", fontSize = 20.sp)
        }
    }
}

@Composable
fun HeroAqiCard(data: AqiUiState.Success) {
    val pred = data.prediction
    val statusColor = parseHexColor(pred.color)

    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = CardBg,
        shape = RoundedCornerShape(24.dp),
        border = androidx.compose.foundation.BorderStroke(2.dp, statusColor)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = data.locationName,
                color = TextPrimary,
                fontSize = 18.sp,
                fontWeight = FontWeight.Black
            )
            Text(
                text = "Updated: ${pred.timestamp.replace("T", " ").take(16)}",
                color = TextSecondary,
                fontSize = 11.sp
            )

            Spacer(modifier = Modifier.height(12.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(text = pred.emoji, fontSize = 42.sp)
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = pred.aqi.toString(),
                    color = statusColor,
                    fontSize = 54.sp,
                    fontWeight = FontWeight.Black
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "AQI",
                    color = TextSecondary,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium
                )
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Category pill
            Surface(
                color = statusColor.copy(alpha = 0.15f),
                shape = CircleShape,
                border = androidx.compose.foundation.BorderStroke(1.dp, statusColor.copy(alpha = 0.4f))
            ) {
                Text(
                    text = pred.level,
                    color = statusColor,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp)
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Scale Bar
            LinearProgressIndicator(
                progress = { (pred.aqi / 500f).coerceIn(0f, 1f) },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(CircleShape),
                color = statusColor,
                trackColor = CardStroke
            )
        }
    }
}

@Composable
fun HealthAdvisoryCard(message: String) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = CardBg,
        shape = RoundedCornerShape(16.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, CardStroke)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Health Recommendation",
                color = AccentBlue,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = message,
                color = TextPrimary,
                fontSize = 13.sp,
                lineHeight = 18.sp
            )
        }
    }
}

@Composable
fun PollutantGrid(pollutants: List<PollutantItem>) {
    Column {
        val pairs = pollutants.chunked(2)
        pairs.forEach { pair ->
            Row(modifier = Modifier.fillMaxWidth()) {
                PollutantCardItem(
                    pollutant = pair[0],
                    modifier = Modifier
                        .weight(1f)
                        .padding(4.dp)
                )
                if (pair.size > 1) {
                    PollutantCardItem(
                        pollutant = pair[1],
                        modifier = Modifier
                            .weight(1f)
                            .padding(4.dp)
                    )
                } else {
                    Spacer(modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
fun PollutantCardItem(pollutant: PollutantItem, modifier: Modifier = Modifier) {
    val statusColor = parseHexColor(pollutant.statusColorHex)

    Surface(
        modifier = modifier,
        color = CardBg,
        shape = RoundedCornerShape(16.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, CardStroke)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = pollutant.key,
                    color = AccentBlue,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
                Surface(
                    color = statusColor.copy(alpha = 0.15f),
                    shape = CircleShape
                ) {
                    Text(
                        text = pollutant.category,
                        color = statusColor,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                    )
                }
            }

            Text(
                text = pollutant.name,
                color = TextSecondary,
                fontSize = 11.sp,
                modifier = Modifier.padding(top = 4.dp),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )

            Row(
                modifier = Modifier.padding(top = 8.dp),
                verticalAlignment = Alignment.Bottom
            ) {
                Text(
                    text = pollutant.formattedValue,
                    color = TextPrimary,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Black
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = pollutant.unit,
                    color = TextSecondary,
                    fontSize = 11.sp,
                    modifier = Modifier.padding(bottom = 2.dp)
                )
            }
        }
    }
}

@Composable
fun ForecastSection(
    currentTab: Int,
    forecastList: List<ForecastItem>,
    isLoading: Boolean,
    onTabSelected: (Int) -> Unit
) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "AQI Predictions & Forecasts",
                color = TextPrimary,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )

            Row {
                FilterChip(
                    selected = currentTab == 0,
                    onClick = { onTabSelected(0) },
                    label = { Text("24-Hour", fontSize = 11.sp) },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = AccentBlue,
                        selectedLabelColor = Color.White
                    )
                )
                Spacer(modifier = Modifier.width(6.dp))
                FilterChip(
                    selected = currentTab == 1,
                    onClick = { onTabSelected(1) },
                    label = { Text("7-Day", fontSize = 11.sp) },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = AccentBlue,
                        selectedLabelColor = Color.White
                    )
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        if (isLoading) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(100.dp),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(
                    modifier = Modifier.size(28.dp),
                    color = AccentBlue,
                    strokeWidth = 3.dp
                )
            }
        } else if (forecastList.isEmpty()) {
            Surface(
                modifier = Modifier.fillMaxWidth(),
                color = CardBg,
                shape = RoundedCornerShape(16.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, CardStroke)
            ) {
                Box(
                    modifier = Modifier.padding(16.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "No forecast data available.",
                        color = TextSecondary,
                        fontSize = 12.sp
                    )
                }
            }
        } else {
            LazyRow(
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                items(forecastList) { forecast ->
                    ForecastCardItem(forecast = forecast)
                }
            }
        }
    }
}

@Composable
fun ForecastCardItem(forecast: ForecastItem) {
    val statusColor = parseHexColor(forecast.color)

    Surface(
        modifier = Modifier.width(125.dp),
        color = CardBg,
        shape = RoundedCornerShape(16.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, CardStroke)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(text = forecast.time, color = TextSecondary, fontSize = 12.sp)
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = forecast.emoji, fontSize = 24.sp)
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "${forecast.aqi} AQI",
                color = statusColor,
                fontSize = 18.sp,
                fontWeight = FontWeight.Black
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = forecast.level,
                color = statusColor,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
fun CustomLocationDialog(
    onDismiss: () -> Unit,
    onConfirm: (Double, Double) -> Unit
) {
    var latText by remember { mutableStateOf("") }
    var lonText by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = CardBg,
        title = {
            Text("Custom Coordinates", color = TextPrimary, fontWeight = FontWeight.Bold)
        },
        text = {
            Column {
                Text(
                    "Enter Latitude and Longitude to query air quality anywhere worldwide.",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedTextField(
                    value = latText,
                    onValueChange = { latText = it },
                    label = { Text("Latitude (e.g. 22.5726)") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = AccentBlue,
                        focusedLabelColor = AccentBlue,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary
                    )
                )
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = lonText,
                    onValueChange = { lonText = it },
                    label = { Text("Longitude (e.g. 88.3639)") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = AccentBlue,
                        focusedLabelColor = AccentBlue,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary
                    )
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    val lat = latText.toDoubleOrNull()
                    val lon = lonText.toDoubleOrNull()
                    if (lat != null && lon != null) {
                        onConfirm(lat, lon)
                    }
                },
                colors = ButtonDefaults.buttonColors(containerColor = AccentBlue)
            ) {
                Text("Search", color = Color.White)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = TextSecondary)
            }
        }
    )
}

private fun parseHexColor(hex: String): Color {
    return try {
        Color(android.graphics.Color.parseColor(hex))
    } catch (e: Exception) {
        AccentBlue
    }
}

@Composable
fun ThreeStepLoadingScreen(loadingState: AqiUiState.Loading) {
    val step = loadingState.step
    val message = loadingState.message

    val animatedProgress by androidx.compose.animation.core.animateFloatAsState(
        targetValue = when (step) {
            1 -> 0.33f
            2 -> 0.66f
            else -> 1.0f
        },
        label = "ProgressAnimation"
    )

    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth(0.92f)
                .border(1.dp, CardStroke, RoundedCornerShape(24.dp)),
            colors = CardDefaults.cardColors(containerColor = CardBg),
            shape = RoundedCornerShape(24.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Top Header Badge
                Text(
                    text = "🍃 AQI VISION ENGINE",
                    color = AccentBlue,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.2.sp
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Progress Bar
                LinearProgressIndicator(
                    progress = { animatedProgress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp)),
                    color = AccentGreen,
                    trackColor = CardStroke
                )

                Spacer(modifier = Modifier.height(24.dp))

                // 3 Step Dots & Indicators
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    StepItem(stepNumber = 1, title = "Open-Meteo", currentStep = step)
                    HorizontalDivider(
                        modifier = Modifier
                            .weight(1f)
                            .height(2.dp),
                        color = if (step >= 2) AccentGreen else CardStroke
                    )
                    StepItem(stepNumber = 2, title = "XGBoost ML", currentStep = step)
                    HorizontalDivider(
                        modifier = Modifier
                            .weight(1f)
                            .height(2.dp),
                        color = if (step >= 3) AccentGreen else CardStroke
                    )
                    StepItem(stepNumber = 3, title = "Forecast", currentStep = step)
                }

                Spacer(modifier = Modifier.height(28.dp))

                // Pulsing Circular Progress
                CircularProgressIndicator(
                    modifier = Modifier.size(48.dp),
                    color = if (step == 1) AccentBlue else if (step == 2) AqiModerate else AccentGreen,
                    strokeWidth = 4.dp
                )

                Spacer(modifier = Modifier.height(20.dp))

                // Step Status Message
                Text(
                    text = message,
                    color = TextPrimary,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium,
                    textAlign = TextAlign.Center,
                    lineHeight = 20.sp
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Step $step of 3",
                    color = TextSecondary,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }
    }
}

@Composable
private fun StepItem(stepNumber: Int, title: String, currentStep: Int) {
    val isCompleted = currentStep > stepNumber
    val isCurrent = currentStep == stepNumber
    val circleColor = when {
        isCompleted -> AccentGreen
        isCurrent -> AccentBlue
        else -> CardStroke
    }
    val textColor = when {
        isCompleted || isCurrent -> TextPrimary
        else -> TextSecondary
    }

    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            modifier = Modifier
                .size(32.dp)
                .background(circleColor, CircleShape),
            contentAlignment = Alignment.Center
        ) {
            if (isCompleted) {
                Text("✓", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
            } else {
                Text("$stepNumber", color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Bold)
            }
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text(text = title, color = textColor, fontSize = 10.sp, fontWeight = FontWeight.Medium)
    }
}


