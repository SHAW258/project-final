package com.aqi_prediction

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.aqi_prediction.presentation.ui.AQIScreen
import com.aqi_prediction.presentation.viewmodel.AqiViewModel
import com.aqi_prediction.presentation.ui.theme.AQIVisionTheme
import dagger.hilt.android.AndroidEntryPoint
import java.util.Locale
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    private val viewModel: AqiViewModel by viewModels()

    @Inject
    lateinit var fusedLocationClient: FusedLocationProviderClient

    private val locationPermissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val fineLocationGranted = permissions[Manifest.permission.ACCESS_FINE_LOCATION] ?: false
        val coarseLocationGranted = permissions[Manifest.permission.ACCESS_COARSE_LOCATION] ?: false

        if (fineLocationGranted || coarseLocationGranted) {
            fetchGpsLocation()
        } else {
            Toast.makeText(this, "Location permission denied. Select a city from the dropdown.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            AQIVisionTheme {
                AQIScreen(
                    viewModel = viewModel,
                    onGpsClick = { checkLocationPermissionsAndFetch() }
                )
            }
        }
    }

    private fun checkLocationPermissionsAndFetch() {
        val finePermission = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
        val coarsePermission = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION)

        if (finePermission == PackageManager.PERMISSION_GRANTED || coarsePermission == PackageManager.PERMISSION_GRANTED) {
            fetchGpsLocation()
        } else {
            locationPermissionRequest.launch(
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                )
            )
        }
    }

    private fun fetchGpsLocation() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
        ) {
            Toast.makeText(this, "Fetching current GPS coordinates…", Toast.LENGTH_SHORT).show()
            fusedLocationClient.lastLocation.addOnSuccessListener { location ->
                if (location != null) {
                    val label = String.format(Locale.US, "GPS (%.4f, %.4f)", location.latitude, location.longitude)
                    viewModel.loadDataForCoordinates(location.latitude, location.longitude, label)
                } else {
                    Toast.makeText(this, "Unable to detect GPS position. Ensure Location Services are enabled.", Toast.LENGTH_LONG).show()
                }
            }.addOnFailureListener {
                Toast.makeText(this, "GPS Location error: ${it.localizedMessage}", Toast.LENGTH_SHORT).show()
            }
        }
    }
}

