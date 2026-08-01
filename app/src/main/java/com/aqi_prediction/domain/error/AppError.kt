package com.aqi_prediction.domain.error

/**
 * Structured application error taxonomy in domain/error package.
 */
sealed class AppError(open val errorCode: String, open val errorMessage: String) {

    // Class A: Network connectivity / timeout errors
    data class NetworkError(
        override val errorCode: String = "ERR_NET_101",
        override val errorMessage: String = "Network connection failed. Please check internet connectivity."
    ) : AppError(errorCode, errorMessage)

    // Class B: Remote API response / HTTP status errors
    data class ApiError(
        val httpCode: Int = 500,
        override val errorCode: String = "ERR_API_202",
        override val errorMessage: String = "Remote server returned invalid or empty payload (HTTP $httpCode)."
    ) : AppError(errorCode, errorMessage)

    // Class C: Machine Learning Backend / XGBoost Model errors
    data class MlBackendError(
        override val errorCode: String = "ERR_ML_303",
        override val errorMessage: String = "XGBoost ML backend model prediction failed or server waking up."
    ) : AppError(errorCode, errorMessage)

    // Class D: Device GPS Location / Permission errors
    data class LocationError(
        override val errorCode: String = "ERR_LOC_404",
        override val errorMessage: String = "Unable to retrieve device GPS location. Ensure location services are enabled."
    ) : AppError(errorCode, errorMessage)

    // Generic Unknown / Unexpected exception fallback
    data class UnknownError(
        val causeMsg: String = "An unexpected error occurred.",
        override val errorCode: String = "ERR_UNK_500",
        override val errorMessage: String = causeMsg
    ) : AppError(errorCode, errorMessage)
}
