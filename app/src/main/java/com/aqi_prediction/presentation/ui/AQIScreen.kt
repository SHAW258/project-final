package com.aqi_prediction.presentation.ui

import android.annotation.SuppressLint
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
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
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aqi_prediction.domain.model.CityLocation
import com.aqi_prediction.domain.model.ForecastItem
import com.aqi_prediction.domain.model.PollutantItem
import com.aqi_prediction.domain.model.PollutantDetailInfo
import com.aqi_prediction.domain.model.PollutantKnowledgeBase
import com.aqi_prediction.presentation.state.AqiUiState
import com.aqi_prediction.presentation.viewmodel.AqiViewModel
import com.aqi_prediction.presentation.ui.theme.*
import androidx.core.graphics.ColorUtils
import androidx.core.graphics.toColorInt
import java.util.Locale

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
    var selectedMainNavTab by remember { mutableStateOf(0) } // 0: Dashboard, 1: Guide, 2: Scale, 3: Health, 4: Team
    var selectedPollutantDetail by remember { mutableStateOf<PollutantDetailInfo?>(null) }

    // Scroll state tracking for Gmail-style auto-hiding top bar
    val mainScrollState = rememberScrollState()
    var isTopBarVisible by remember { mutableStateOf(true) }
    var previousScrollValue by remember { mutableIntStateOf(0) }

    LaunchedEffect(mainScrollState.value) {
        val currentScroll = mainScrollState.value
        val delta = currentScroll - previousScrollValue
        if (currentScroll <= 15) {
            // At or near top -> always show top bar
            isTopBarVisible = true
        } else if (delta > 10) {
            // Scrolling down -> hide top bar smoothly like Gmail
            isTopBarVisible = false
        } else if (delta < -10) {
            // Scrolling up -> show top bar smoothly like Gmail
            isTopBarVisible = true
        }
        previousScrollValue = currentScroll
    }

    // Reset top bar and scroll when changing main navigation tabs
    LaunchedEffect(selectedMainNavTab) {
        isTopBarVisible = true
        mainScrollState.scrollTo(0)
        previousScrollValue = 0
    }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        floatingActionButton = {
            FloatingActionButton(
                onClick = { viewModel.refreshCurrentLocation() },
                containerColor = MaterialTheme.colorScheme.primary,
                contentColor = Color.White,
                shape = CircleShape
            ) {
                Icon(
                    imageVector = AppIcons.REFRESH,
                    contentDescription = "Refresh",
                    modifier = Modifier.size(24.dp)
                )
            }
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
        ) {
            // Collapsible Top Header & Location Bar (Gmail-style hide on scroll down)
            AnimatedVisibility(
                visible = isTopBarVisible,
                enter = androidx.compose.animation.expandVertically() + androidx.compose.animation.fadeIn(),
                exit = androidx.compose.animation.shrinkVertically() + androidx.compose.animation.fadeOut()
            ) {
                Column {
                    HeaderSection(
                        isOnline = isBackendOnline,
                        onCheckHealth = { viewModel.checkBackendHealth() }
                    )

                    Spacer(modifier = Modifier.height(4.dp))

                    LocationSelectorBar(
                        presetCities = viewModel.presetCities,
                        selectedCity = selectedCity,
                        onCitySelected = { viewModel.loadDataForCity(it) },
                        onGpsClick = onGpsClick,
                        onCustomCoordsClick = { showCustomDialog = true }
                    )

                    Spacer(modifier = Modifier.height(8.dp))
                }
            }

            // Main Navigation Bar Tabs (Dashboard, Guide, Scale, Advice, Team)
            MainNavigationTabs(
                selectedTab = selectedMainNavTab,
                onTabSelected = { selectedMainNavTab = it }
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Body Switcher based on Selected Navigation Tab
            when (selectedMainNavTab) {
                0 -> {
                    // Content Body State for Dashboard
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
                                        .border(1.dp, AQI_UNHEALTHY, RoundedCornerShape(20.dp)),
                                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                                    shape = RoundedCornerShape(20.dp)
                                ) {
                                    Column(
                                        modifier = Modifier.padding(24.dp),
                                        horizontalAlignment = Alignment.CenterHorizontally
                                    ) {
                                        Icon(
                                            imageVector = AppIcons.WARNING,
                                            contentDescription = null,
                                            tint = AQI_UNHEALTHY,
                                            modifier = Modifier.size(48.dp)
                                        )
                                        Spacer(modifier = Modifier.height(12.dp))
                                        Text(
                                            text = "ERROR [${err.errorCode}]",
                                            color = AQI_UNHEALTHY,
                                            fontSize = 18.sp,
                                            fontWeight = FontWeight.Bold,
                                            letterSpacing = 1.sp
                                        )
                                        Spacer(modifier = Modifier.height(12.dp))
                                        Text(
                                            text = err.errorMessage,
                                            color = MaterialTheme.colorScheme.onSurface,
                                            fontSize = 14.sp,
                                            textAlign = TextAlign.Center,
                                            lineHeight = 20.sp
                                        )
                                        Spacer(modifier = Modifier.height(20.dp))
                                        Button(
                                            onClick = { viewModel.refreshCurrentLocation() },
                                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
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
                                    .verticalScroll(mainScrollState)
                                    .padding(bottom = 80.dp)
                            ) {
                                // Hero AQI Card
                                HeroAqiCard(data = state)

                                Spacer(modifier = Modifier.height(14.dp))

                                // Health Advisory Card
                                HealthAdvisoryCard(message = state.prediction.healthMessage)

                                Spacer(modifier = Modifier.height(16.dp))

                                // Live Pollutants Section Header
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = AppStrings.LIVE_POLLUTANTS,
                                        color = MaterialTheme.colorScheme.onSurface,
                                        fontSize = 16.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                    Text(
                                        text = "💡 Tap item for guide",
                                        color = MaterialTheme.colorScheme.primary,
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.Medium
                                    )
                                }

                                Spacer(modifier = Modifier.height(8.dp))

                                PollutantGrid(
                                    pollutants = state.pollutants,
                                    onPollutantClick = { item ->
                                        selectedPollutantDetail = PollutantKnowledgeBase.pollutantDetailsMap[item.key]
                                    }
                                )

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
                1 -> {
                    // Pollutants Guide Tab
                    PollutantsGuideScreen(
                        scrollState = mainScrollState,
                        onSelectPollutant = { detail -> selectedPollutantDetail = detail }
                    )
                }
                2 -> {
                    // AQI Scale Guide Tab
                    AqiScaleGuideScreen(scrollState = mainScrollState)
                }
                3 -> {
                    // Health Advice Tab
                    HealthAdvisoriesScreen(scrollState = mainScrollState)
                }
                4 -> {
                    // About Team Tab
                    AboutTeamScreen(scrollState = mainScrollState)
                }
            }
        }
    }

    if (showCustomDialog) {
        CustomLocationDialog(
            onDismiss = { showCustomDialog = false },
            onConfirm = { lat, lon ->
                showCustomDialog = false
                val label = String.format(Locale.US, "Coords (%.2f, %.2f)", lat, lon)
                viewModel.loadDataForCoordinates(lat, lon, label)
            }
        )
    }

    selectedPollutantDetail?.let { detail ->
        PollutantDetailModalDialog(
            detail = detail,
            onDismiss = { selectedPollutantDetail = null }
        )
    }
}

@Composable
fun MainNavigationTabs(
    selectedTab: Int,
    onTabSelected: (Int) -> Unit
) {
    val tabs = listOf(
        "📊 Dashboard",
        "📖 Pollutants",
        "🛡️ AQI Scale",
        "🩺 Health",
        "👥 Team"
    )

    ScrollableTabRow(
        selectedTabIndex = selectedTab,
        edgePadding = 0.dp,
        containerColor = MaterialTheme.colorScheme.surface,
        contentColor = MaterialTheme.colorScheme.primary,
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .border(1.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(12.dp))
    ) {
        tabs.forEachIndexed { index, title ->
            Tab(
                selected = selectedTab == index,
                onClick = { onTabSelected(index) },
                text = {
                    Text(
                        text = title,
                        fontSize = 12.sp,
                        fontWeight = if (selectedTab == index) FontWeight.Bold else FontWeight.Normal,
                        color = if (selectedTab == index) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            )
        }
    }
}

@Composable
fun HeaderSection(isOnline: Boolean, onCheckHealth: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "🍃",
            fontSize = 28.sp
        )
        Spacer(modifier = Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = AppStrings.APP_NAME,
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 20.sp,
                fontWeight = FontWeight.Black
            )
            Text(
                text = AppStrings.APP_SLOGAN,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 11.sp
            )
        }
        // Online status pill
        Surface(
            onClick = onCheckHealth,
            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
            shape = CircleShape,
            border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.25f))
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .background(if (isOnline) STATUS_ONLINE else STATUS_OFFLINE, CircleShape)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = if (isOnline) AppStrings.BACKEND_ONLINE else AppStrings.BACKEND_OFFLINE,
                    color = MaterialTheme.colorScheme.onSurface,
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
            color = MaterialTheme.colorScheme.surface,
            shape = RoundedCornerShape(12.dp),
            border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
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
                    color = MaterialTheme.colorScheme.onSurface,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(" 🔻", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }

            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false },
                modifier = Modifier.background(MaterialTheme.colorScheme.surface)
            ) {
                presetCities.forEach { city ->
                    DropdownMenuItem(
                        text = { Text("${city.name}, ${city.country}", color = MaterialTheme.colorScheme.onSurface) },
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
                .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(12.dp))
                .border(1.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(12.dp))
        ) {
            Icon(
                imageVector = AppIcons.GPS,
                contentDescription = "GPS",
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(24.dp)
            )
        }

        Spacer(modifier = Modifier.width(8.dp))

        // Custom Location Button
        IconButton(
            onClick = onCustomCoordsClick,
            modifier = Modifier
                .size(44.dp)
                .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(12.dp))
                .border(1.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(12.dp))
        ) {
            Icon(
                imageVector = AppIcons.LOCATION,
                contentDescription = "Custom Location",
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(24.dp)
            )
        }
    }
}

@Composable
fun HeroAqiCard(data: AqiUiState.Success) {
    val pred = data.prediction
    val statusColor = parseHexColor(pred.color)

    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surface,
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
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 18.sp,
                fontWeight = FontWeight.Black
            )
            Text(
                text = "Updated: ${pred.timestamp.replace("T", " ").take(16)}",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
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
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium
                )
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Category pill matching chart screenshot style
            val upperHex = pred.color.uppercase()
            val pillTextColor = if (upperHex.contains("98EC") || upperHex.contains("FFC0") || upperHex.contains("00E4") || upperHex.contains("FFFF00")) Color.Black else Color.White

            Surface(
                color = statusColor,
                shape = CircleShape,
                border = androidx.compose.foundation.BorderStroke(1.dp, Color.Black.copy(alpha = 0.25f))
            ) {
                Text(
                    text = pred.level,
                    color = pillTextColor,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Black,
                    modifier = Modifier.padding(horizontal = 18.dp, vertical = 6.dp)
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
                trackColor = MaterialTheme.colorScheme.outline
            )
        }
    }
}

@Composable
fun HealthAdvisoryCard(message: String) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(16.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = AppIcons.INFO,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Health Recommendation",
                    color = MaterialTheme.colorScheme.primary,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = message,
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 13.sp,
                lineHeight = 18.sp
            )
        }
    }
}

@Composable
fun PollutantGrid(
    pollutants: List<PollutantItem>,
    onPollutantClick: (PollutantItem) -> Unit
) {
    Column {
        val pairs = pollutants.chunked(2)
        pairs.forEach { pair ->
            Row(modifier = Modifier.fillMaxWidth()) {
                PollutantCardItem(
                    pollutant = pair[0],
                    modifier = Modifier
                        .weight(1f)
                        .padding(4.dp),
                    onClick = { onPollutantClick(pair[0]) }
                )
                if (pair.size > 1) {
                    PollutantCardItem(
                        pollutant = pair[1],
                        modifier = Modifier
                            .weight(1f)
                            .padding(4.dp),
                        onClick = { onPollutantClick(pair[1]) }
                    )
                } else {
                    Spacer(modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
fun PollutantCardItem(
    pollutant: PollutantItem,
    modifier: Modifier = Modifier,
    onClick: () -> Unit = {}
) {
    val statusColor = parseHexColor(pollutant.statusColorHex)

    Surface(
        modifier = modifier.clickable { onClick() },
        color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(16.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = pollutant.key,
                    color = MaterialTheme.colorScheme.primary,
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
                color = MaterialTheme.colorScheme.onSurfaceVariant,
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
                    color = MaterialTheme.colorScheme.onSurface,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Black
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = pollutant.unit,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
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
                text = AppStrings.FORECAST_HEADER,
                color = MaterialTheme.colorScheme.onSurface,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )

            Row {
                FilterChip(
                    selected = currentTab == 0,
                    onClick = { onTabSelected(0) },
                    label = { Text("24-Hour", fontSize = 11.sp) },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = MaterialTheme.colorScheme.primary,
                        selectedLabelColor = Color.White
                    )
                )
                Spacer(modifier = Modifier.width(6.dp))
                FilterChip(
                    selected = currentTab == 1,
                    onClick = { onTabSelected(1) },
                    label = { Text("7-Day", fontSize = 11.sp) },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = MaterialTheme.colorScheme.primary,
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
                    color = MaterialTheme.colorScheme.primary,
                    strokeWidth = 3.dp
                )
            }
        } else if (forecastList.isEmpty()) {
            Surface(
                modifier = Modifier.fillMaxWidth(),
                color = MaterialTheme.colorScheme.surface,
                shape = RoundedCornerShape(16.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
            ) {
                Box(
                    modifier = Modifier.padding(16.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = AppStrings.NO_FORECAST,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
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
        color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(16.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(text = forecast.time, color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 12.sp)
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

/* ========================================================================= */
/* MERGED FRONTEND FEATURES: Pollutant Guide Screen                          */
/* ========================================================================= */
@Composable
fun PollutantsGuideScreen(
    scrollState: androidx.compose.foundation.ScrollState = rememberScrollState(),
    onSelectPollutant: (PollutantDetailInfo) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(bottom = 80.dp)
    ) {
        Text(
            text = "📖 Comprehensive Pollutant Directory",
            color = MaterialTheme.colorScheme.onSurface,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Learn about key air pollutants, healthy target ranges, EPA standard safety thresholds, and health risks.",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontSize = 12.sp,
            modifier = Modifier.padding(top = 4.dp, bottom = 12.dp)
        )

        PollutantKnowledgeBase.pollutantDetailsMap.values.forEach { detail ->
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 6.dp)
                    .clickable { onSelectPollutant(detail) },
                color = MaterialTheme.colorScheme.surface,
                shape = RoundedCornerShape(16.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(text = detail.icon, fontSize = 24.sp)
                            Spacer(modifier = Modifier.width(10.dp))
                            Column {
                                Text(
                                    text = detail.key,
                                    color = MaterialTheme.colorScheme.primary,
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = detail.name,
                                    color = MaterialTheme.colorScheme.onSurface,
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }
                        Surface(
                            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                            shape = CircleShape
                        ) {
                            Text(
                                text = "Target: ${detail.healthyRange}",
                                color = MaterialTheme.colorScheme.primary,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))
                    Text(
                        text = detail.description,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontSize = 12.sp,
                        lineHeight = 16.sp
                    )

                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "⚡ Primary Sources:",
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                    detail.sources.take(2).forEach { source ->
                        Text(
                            text = "  • $source",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 11.sp
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.End
                    ) {
                        Text(
                            text = "Tap to view full details →",
                            color = MaterialTheme.colorScheme.primary,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }
}

/* ========================================================================= */
/* MERGED FRONTEND FEATURES: AQI Scale Guide Screen (Matching Chart Screenshot)*/
/* ========================================================================= */
@Composable
fun AqiScaleGuideScreen(
    scrollState: androidx.compose.foundation.ScrollState = rememberScrollState()
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(bottom = 80.dp)
    ) {
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = MaterialTheme.colorScheme.surface,
            shape = RoundedCornerShape(16.dp),
            border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "AIR QUALITY INDEX GUIDE",
                    color = MaterialTheme.colorScheme.onSurface,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Black,
                    letterSpacing = 1.sp,
                    textAlign = TextAlign.Center
                )

                Spacer(modifier = Modifier.height(12.dp))

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "AQI Colors",
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Range",
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                HorizontalDivider(color = MaterialTheme.colorScheme.outline, thickness = 1.5.dp)

                Spacer(modifier = Modifier.height(8.dp))

                PollutantKnowledgeBase.aqiLevelsList.forEach { tier ->
                    val bgColor = parseHexColor(tier.colorHex)
                    val txtColor = if (tier.textColorHex.equals("#000000", ignoreCase = true)) Color.Black else Color.White

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp)
                    ) {
                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            color = bgColor,
                            shape = RoundedCornerShape(8.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, Color.Black.copy(alpha = 0.2f))
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 16.dp, vertical = 14.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = tier.level,
                                    color = txtColor,
                                    fontSize = 15.sp,
                                    fontWeight = FontWeight.Black,
                                    modifier = Modifier.weight(1f)
                                )

                                Text(
                                    text = tier.range,
                                    color = txtColor,
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Black
                                )
                            }
                        }

                        // Guidance detail below row
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 2.dp),
                            color = MaterialTheme.colorScheme.surface,
                            shape = RoundedCornerShape(bottomStart = 8.dp, bottomEnd = 8.dp)
                        ) {
                            Column(modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)) {
                                Text(
                                    text = tier.description,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    fontSize = 11.sp,
                                    lineHeight = 15.sp
                                )
                                Text(
                                    text = "💡 ${tier.healthAdvice}",
                                    color = MaterialTheme.colorScheme.primary,
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.SemiBold,
                                    modifier = Modifier.padding(top = 2.dp)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ========================================================================= */
/* MERGED FRONTEND FEATURES: Health Advisories Screen                         */
/* ========================================================================= */
@Composable
fun HealthAdvisoriesScreen(
    scrollState: androidx.compose.foundation.ScrollState = rememberScrollState()
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(bottom = 80.dp)
    ) {
        Text(
            text = "🩺 Sensitive Groups & Health Safety",
            color = MaterialTheme.colorScheme.onSurface,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Targeted precautions for vulnerable populations and high-risk conditions.",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontSize = 12.sp,
            modifier = Modifier.padding(top = 4.dp, bottom = 12.dp)
        )

        PollutantKnowledgeBase.sensitiveGroupsAdviceList.forEach { group ->
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 6.dp),
                color = MaterialTheme.colorScheme.surface,
                shape = RoundedCornerShape(16.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(text = group.icon, fontSize = 26.sp)
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            text = group.title,
                            color = MaterialTheme.colorScheme.primary,
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = group.summary,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 12.sp,
                        lineHeight = 16.sp
                    )

                    Spacer(modifier = Modifier.height(10.dp))
                    Text(
                        text = "Safety Action Plan:",
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )

                    group.adviceList.forEach { advice ->
                        Row(
                            modifier = Modifier.padding(vertical = 2.dp),
                            verticalAlignment = Alignment.Top
                        ) {
                            Text(text = "✅ ", fontSize = 11.sp)
                            Text(
                                text = advice,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                fontSize = 11.sp,
                                lineHeight = 15.sp
                            )
                        }
                    }
                }
            }
        }
    }
}

/* ========================================================================= */
/* MERGED FRONTEND FEATURES: About Team Screen                                */
/* ========================================================================= */
@Composable
fun AboutTeamScreen(
    scrollState: androidx.compose.foundation.ScrollState = rememberScrollState()
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(bottom = 80.dp)
    ) {
        Text(
            text = "👥 AQI Prediction Project Team",
            color = MaterialTheme.colorScheme.onSurface,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Final Year B.Tech Computer Science & Engineering Project at Seacom Engineering College.",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontSize = 12.sp,
            modifier = Modifier.padding(top = 4.dp, bottom = 12.dp)
        )

        PollutantKnowledgeBase.projectTeamMembers.forEach { member ->
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 6.dp),
                color = MaterialTheme.colorScheme.surface,
                shape = RoundedCornerShape(16.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = member["name"] ?: "",
                        color = MaterialTheme.colorScheme.primary,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Black
                    )
                    Text(
                        text = member["role"] ?: "",
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "${member["college"]} • ${member["specialization"]}",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontSize = 11.sp,
                        modifier = Modifier.padding(top = 2.dp)
                    )

                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Key Contributions: ${member["contributions"]}",
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }
    }
}

/* ========================================================================= */
/* MERGED FRONTEND FEATURES: Pollutant Detail Dialog / Modal                 */
/* ========================================================================= */
@Composable
fun PollutantDetailModalDialog(
    detail: PollutantDetailInfo,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = MaterialTheme.colorScheme.surface,
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(text = detail.icon, fontSize = 28.sp)
                Spacer(modifier = Modifier.width(10.dp))
                Column {
                    Text(
                        text = detail.key,
                        color = MaterialTheme.colorScheme.primary,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Black
                    )
                    Text(
                        text = detail.name,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = detail.description,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontSize = 12.sp,
                    lineHeight = 17.sp
                )

                Spacer(modifier = Modifier.height(12.dp))

                // Healthy Range & Danger Level
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Surface(
                        color = ACCENT_GREEN.copy(alpha = 0.15f),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.weight(1f)
                    ) {
                        Column(modifier = Modifier.padding(8.dp)) {
                            Text("Healthy Target", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text(detail.healthyRange, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = ACCENT_GREEN)
                        }
                    }

                    Spacer(modifier = Modifier.width(8.dp))

                    Surface(
                        color = AQI_UNHEALTHY.copy(alpha = 0.15f),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.weight(1f)
                    ) {
                        Column(modifier = Modifier.padding(8.dp)) {
                            Text("Danger Threshold", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text(detail.dangerLevel, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = AQI_UNHEALTHY)
                        }
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                Text("🏛️ EPA Standard:", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurface)
                Text(detail.epaStandard, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)

                Spacer(modifier = Modifier.height(12.dp))

                Text("🏭 Major Pollution Sources:", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurface)
                detail.sources.forEach { source ->
                    Text(" • $source", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }

                Spacer(modifier = Modifier.height(12.dp))

                Text("🫁 Health Consequences:", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurface)
                detail.healthEffects.forEach { effect ->
                    Text(" • $effect", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        },
        confirmButton = {
            Button(
                onClick = onDismiss,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text("Got It", color = Color.White)
            }
        }
    )
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
        containerColor = MaterialTheme.colorScheme.surface,
        title = {
            Text(AppStrings.CUSTOM_LOCATION_TITLE, color = MaterialTheme.colorScheme.onSurface, fontWeight = FontWeight.Bold)
        },
        text = {
            Column {
                Text(
                    AppStrings.CUSTOM_LOCATION_DESC,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
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
                        focusedBorderColor = MaterialTheme.colorScheme.primary,
                        focusedLabelColor = MaterialTheme.colorScheme.primary,
                        focusedTextColor = MaterialTheme.colorScheme.onSurface,
                        unfocusedTextColor = MaterialTheme.colorScheme.onSurface
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
                        focusedBorderColor = MaterialTheme.colorScheme.primary,
                        focusedLabelColor = MaterialTheme.colorScheme.primary,
                        focusedTextColor = MaterialTheme.colorScheme.onSurface,
                        unfocusedTextColor = MaterialTheme.colorScheme.onSurface
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
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text("Search", color = Color.White)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    )
}

@SuppressLint("UseKtx")
@Composable
private fun parseHexColor(hex: String): Color {
    return when (hex.uppercase().trim()) {
        "#98EC85", "#00E400" -> AQI_GOOD
        "#FFC000", "#FFFF00", "#FFC107" -> AQI_MODERATE
        "#FF7E00" -> AQI_SENSITIVE
        "#E51A1A", "#FF0000" -> AQI_UNHEALTHY
        "#8F3F97" -> AQI_VERY_UNHEALTHY
        "#660014", "#7E0023" -> AQI_HAZARDOUS
        else -> try {
            Color(hex.toColorInt())
        } catch (e: Exception) {
            ACCENT_BLUE
        }
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
                .border(1.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(24.dp)),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
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
                    color = MaterialTheme.colorScheme.primary,
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
                    color = ACCENT_GREEN,
                    trackColor = MaterialTheme.colorScheme.outline
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
                        color = if (step >= 2) ACCENT_GREEN else MaterialTheme.colorScheme.outline
                    )
                    StepItem(stepNumber = 2, title = "XGBoost ML", currentStep = step)
                    HorizontalDivider(
                        modifier = Modifier
                            .weight(1f)
                            .height(2.dp),
                        color = if (step >= 3) ACCENT_GREEN else MaterialTheme.colorScheme.outline
                    )
                    StepItem(stepNumber = 3, title = "Forecast", currentStep = step)
                }

                Spacer(modifier = Modifier.height(28.dp))

                // Pulsing Circular Progress
                CircularProgressIndicator(
                    modifier = Modifier.size(48.dp),
                    color = if (step == 1) MaterialTheme.colorScheme.primary else if (step == 2) parseHexColor("#FFFF00") else ACCENT_GREEN,
                    strokeWidth = 4.dp
                )

                Spacer(modifier = Modifier.height(20.dp))

                // Step Status Message
                Text(
                    text = message,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium,
                    textAlign = TextAlign.Center,
                    lineHeight = 20.sp
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Step $step of 3",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
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
        isCompleted -> ACCENT_GREEN
        isCurrent -> MaterialTheme.colorScheme.primary
        else -> MaterialTheme.colorScheme.outline
    }
    val textColor = when {
        isCompleted || isCurrent -> MaterialTheme.colorScheme.onSurface
        else -> MaterialTheme.colorScheme.onSurfaceVariant
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
