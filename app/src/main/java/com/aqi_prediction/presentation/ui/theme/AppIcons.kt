package com.aqi_prediction.presentation.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathFillType
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.unit.dp

object AppIcons {
    val REFRESH: ImageVector
        get() = ImageVector.Builder(
            name = "Refresh",
            defaultWidth = 24.dp,
            defaultHeight = 24.dp,
            viewportWidth = 24f,
            viewportHeight = 24f
        ).path(
            fill = SolidColor(Color.Black),
            pathFillType = PathFillType.NonZero
        ) {
            moveTo(17.65f, 6.35f)
            curveTo(16.2f, 4.9f, 14.21f, 4f, 12f, 4f)
            curveTo(7.58f, 4f, 4.01f, 7.58f, 4.01f, 12f)
            reflectiveCurveTo(7.58f, 20f, 12f, 20f)
            curveTo(15.73f, 20f, 18.84f, 17.45f, 19.73f, 14f)
            horizontalLineTo(17.65f)
            curveTo(16.83f, 16.33f, 14.61f, 18f, 12f, 18f)
            curveTo(8.69f, 18f, 6f, 15.31f, 6f, 12f)
            reflectiveCurveTo(8.69f, 6f, 12f, 6f)
            curveTo(13.66f, 6f, 15.14f, 6.69f, 16.22f, 1.78f)
            lineTo(13f, 11f)
            horizontalLineTo(20f)
            verticalLineTo(4f)
            lineTo(17.65f, 6.35f)
            close()
        }.build()

    val GPS: ImageVector
        get() = ImageVector.Builder(
            name = "Gps",
            defaultWidth = 24.dp,
            defaultHeight = 24.dp,
            viewportWidth = 24f,
            viewportHeight = 24f
        ).path(
            fill = SolidColor(Color.Black),
            pathFillType = PathFillType.NonZero
        ) {
            moveTo(12f, 8f)
            curveTo(9.79f, 8f, 8f, 9.79f, 8f, 12f)
            reflectiveCurveTo(9.79f, 16f, 12f, 20f)
            reflectiveCurveTo(16f, 14.21f, 16f, 12f)
            reflectiveCurveTo(14.21f, 8f, 12f, 8f)
            close()
            moveTo(20.94f, 11f)
            curveTo(20.48f, 6.83f, 17.17f, 3.52f, 13f, 3.06f)
            verticalLineTo(1f)
            horizontalLineTo(11f)
            verticalLineTo(3.06f)
            curveTo(6.83f, 3.52f, 3.52f, 6.83f, 3.06f, 11f)
            horizontalLineTo(1f)
            verticalLineTo(13f)
            horizontalLineTo(3.06f)
            curveTo(3.52f, 17.17f, 6.83f, 20.48f, 11f, 20.94f)
            verticalLineTo(23f)
            horizontalLineTo(13f)
            verticalLineTo(20.94f)
            curveTo(17.17f, 20.48f, 20.48f, 17.17f, 20.94f, 13f)
            horizontalLineTo(23f)
            verticalLineTo(11f)
            horizontalLineTo(20.94f)
            close()
            moveTo(12f, 19f)
            curveTo(8.13f, 19f, 5f, 15.87f, 5f, 12f)
            reflectiveCurveTo(8.13f, 5f, 12f, 5f)
            reflectiveCurveTo(19f, 8.13f, 19f, 12f)
            reflectiveCurveTo(15.87f, 19f, 12f, 19f)
            close()
        }.build()

    val LOCATION: ImageVector
        get() = ImageVector.Builder(
            name = "Location",
            defaultWidth = 24.dp,
            defaultHeight = 24.dp,
            viewportWidth = 24f,
            viewportHeight = 24f
        ).path(
            fill = SolidColor(Color.Black),
            pathFillType = PathFillType.NonZero
        ) {
            moveTo(12f, 2f)
            curveTo(8.13f, 2f, 5f, 5.13f, 5f, 9f)
            curveTo(5f, 14.25f, 12f, 22f, 12f, 22f)
            reflectiveCurveTo(19f, 14.25f, 19f, 9f)
            curveTo(19f, 5.13f, 15.87f, 2f, 12f, 2f)
            close()
            moveTo(12f, 11.5f)
            curveTo(10.62f, 11.5f, 9.5f, 10.38f, 9.5f, 9f)
            reflectiveCurveTo(10.62f, 6.5f, 12f, 6.5f)
            reflectiveCurveTo(14.5f, 7.62f, 14.5f, 9f)
            reflectiveCurveTo(13.38f, 11.5f, 12f, 11.5f)
            close()
        }.build()

    val WARNING: ImageVector
        get() = ImageVector.Builder(
            name = "Warning",
            defaultWidth = 24.dp,
            defaultHeight = 24.dp,
            viewportWidth = 24f,
            viewportHeight = 24f
        ).path(
            fill = SolidColor(Color.Black),
            pathFillType = PathFillType.NonZero
        ) {
            moveTo(1f, 21f)
            horizontalLineTo(23f)
            lineTo(12f, 2f)
            lineTo(1f, 21f)
            close()
            moveTo(13f, 18f)
            horizontalLineTo(11f)
            verticalLineTo(16f)
            horizontalLineTo(13f)
            verticalLineTo(18f)
            close()
            moveTo(13f, 14f)
            horizontalLineTo(11f)
            verticalLineTo(10f)
            horizontalLineTo(13f)
            verticalLineTo(14f)
            close()
        }.build()

    val INFO: ImageVector
        get() = ImageVector.Builder(
            name = "Info",
            defaultWidth = 24.dp,
            defaultHeight = 24.dp,
            viewportWidth = 24f,
            viewportHeight = 24f
        ).path(
            fill = SolidColor(Color.Black),
            pathFillType = PathFillType.NonZero
        ) {
            moveTo(12f, 2f)
            curveTo(6.48f, 2f, 2f, 6.48f, 2f, 12f)
            reflectiveCurveTo(6.48f, 22f, 12f, 22f)
            reflectiveCurveTo(22f, 17.52f, 22f, 12f)
            reflectiveCurveTo(17.52f, 2f, 12f, 2f)
            close()
            moveTo(13f, 17f)
            horizontalLineTo(11f)
            verticalLineTo(11f)
            horizontalLineTo(13f)
            verticalLineTo(17f)
            close()
            moveTo(13f, 9f)
            horizontalLineTo(11f)
            verticalLineTo(7f)
            horizontalLineTo(13f)
            verticalLineTo(9f)
            close()
        }.build()
}
