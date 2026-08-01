# AQI Vision 🍃

**AQI Vision** is a professional Android application designed to provide real-time air quality monitoring and machine learning-powered forecasts. Built with modern Android technologies, it offers a sleek user experience for staying informed about the air you breathe.

## 🚀 Features

-   **Real-Time Monitoring**: Get instant updates on the Air Quality Index (AQI) for any location worldwide.
-   **ML-Powered Forecasts**: Uses XGBoost-based machine learning models to provide 24-hour and 7-day AQI predictions.
-   **Pollutant Breakdown**: Detailed analysis of major pollutants including PM2.5, PM10, NO2, SO2, O3, and CO.
-   **Smart Location Detection**: Automatic GPS tracking using Google Play Services for hyper-local data.
-   **Health Advisories**: Personalized recommendations based on current air quality levels to help you stay safe.
-   **Custom Coordinates**: Search for air quality data anywhere on the globe by entering custom latitude and longitude.
-   **Dark Mode UI**: A premium, modern interface designed with Jetpack Compose and Material 3.

## 🛠 Tech Stack

-   **Language**: [Kotlin](https://kotlinlang.org/)
-   **UI Framework**: [Jetpack Compose](https://developer.android.com/jetpack/compose) (Material 3)
-   **Architecture**: MVVM + Clean Architecture (Domain, Data, Presentation layers)
-   **Dependency Injection**: [Dagger Hilt](https://dagger.dev/hilt/)
-   **Networking**: [Retrofit](https://square.github.io/retrofit/) & [OkHttp](https://square.github.io/okhttp/)
-   **Asynchrony**: Kotlin Coroutines & Flow
-   **Location**: Fused Location Provider API
-   **JSON Parsing**: Gson
-   **Build System**: Gradle Kotlin DSL (.kts)

## 📁 Project Structure

```text
com.aqi_prediction
├── core/                # Networking clients and utility classes
├── data/                # API definitions, DTOs, Mappers, and Repository implementations
├── di/                  # Dagger Hilt Modules
├── domain/              # Domain models, Repository interfaces, and Use Cases
└── presentation/        # UI (Compose), State, Events, and ViewModels
```

## ⚙️ Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/AQI_Prediction.git
    ```
2.  **Open in Android Studio**:
    -   Ensure you have the latest version of Android Studio (Ladybug or newer recommended).
3.  **Build Project**:
    -   Wait for Gradle sync to complete.
4.  **Run**:
    -   Deploy to a physical device or emulator with Google Play Services.

## 📡 API Integrations

-   **Open-Meteo API**: Primary source for global air quality and weather data.
-   **ML Backend**: Custom XGBoost integration for advanced predictive analysis.

## 📄 License

This project is licensed under the MIT License.

---
*Developed with ❤️ for a cleaner planet.*
