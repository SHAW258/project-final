package com.aqi_prediction.domain.model

object PollutantKnowledgeBase {

    val pollutantDetailsMap: Map<String, PollutantDetailInfo> = mapOf(
        "PM2.5" to PollutantDetailInfo(
            key = "PM2.5",
            name = "Fine Particulate Matter (PM2.5)",
            icon = "🌫️",
            description = "Tiny invisible inhalable particles smaller than 2.5 micrometers in diameter that can penetrate deep into human lung tissue and bloodstream.",
            healthyRange = "0 - 30 μg/m³",
            dangerLevel = "> 60 μg/m³",
            epaStandard = "Annual: 12.0 μg/m³ | 24-hr: 35 μg/m³",
            sources = listOf(
                "Automotive & Diesel vehicle emissions",
                "Industrial smoke & power plant combustion",
                "Wildfires & agricultural biomass burning",
                "Residential wood heating & cooking fuel"
            ),
            healthEffects = listOf(
                "Decreased lung capacity & respiratory inflammation",
                "Increased risk of cardiovascular disease & heart attacks",
                "Aggravated asthma attacks & chronic bronchitis",
                "Premature mortality in individuals with pre-existing heart/lung issues"
            )
        ),
        "PM10" to PollutantDetailInfo(
            key = "PM10",
            name = "Coarse Particulate Matter (PM10)",
            icon = "💨",
            description = "Inhalable dust and smoke particles between 2.5 and 10 micrometers in diameter.",
            healthyRange = "0 - 50 μg/m³",
            dangerLevel = "> 100 μg/m³",
            epaStandard = "24-hr: 150 μg/m³",
            sources = listOf(
                "Construction site dust & road dust resuspension",
                "Agricultural tilling & open dirt roads",
                "Dust storms & crushing/grinding operations",
                "Industrial processing plants"
            ),
            healthEffects = listOf(
                "Irritation of eyes, nose, throat, and upper airways",
                "Persistent coughing, wheezing, and breathing difficulty",
                "Aggravation of lung disease and asthma symptoms",
                "Reduced outdoor exercise tolerance"
            )
        ),
        "NO2" to PollutantDetailInfo(
            key = "NO2",
            name = "Nitrogen Dioxide (NO2)",
            icon = "🏭",
            description = "A pungent reddish-brown gas formed primarily during high-temperature fuel combustion.",
            healthyRange = "0 - 40 ppb",
            dangerLevel = "> 80 ppb",
            epaStandard = "1-hr: 100 ppb | Annual: 53 ppb",
            sources = listOf(
                "Internal combustion vehicle exhausts",
                "Fossil fuel power plants & heating boilers",
                "Heavy industrial manufacturing facilities",
                "Indoor unvented gas stoves & heaters"
            ),
            healthEffects = listOf(
                "Airway lining irritation and bronchial constriction",
                "Increased susceptibility to respiratory tract infections",
                "Triggering of severe asthma flare-ups",
                "Key precursor to ground-level ozone (smog) formation"
            )
        ),
        "SO2" to PollutantDetailInfo(
            key = "SO2",
            name = "Sulfur Dioxide (SO2)",
            icon = "⚗️",
            description = "A heavy, colorless gas with a sharp, suffocating odor produced by burning sulfurous fossil fuels.",
            healthyRange = "0 - 40 ppb",
            dangerLevel = "> 80 ppb",
            epaStandard = "1-hr: 75 ppb | 24-hr: 140 ppb",
            sources = listOf(
                "Coal-fired power generation plants",
                "Petroleum oil refineries & chemical manufacturing",
                "Metal ore smelting & extraction",
                "Volcanic eruptions & industrial boilers"
            ),
            healthEffects = listOf(
                "Immediate throat irritation & respiratory distress",
                "Severe bronchoconstriction in asthma sufferers",
                "Aggravation of existing cardiovascular & lung conditions",
                "Precursor to acid rain and secondary particulate sulfate"
            )
        ),
        "CO" to PollutantDetailInfo(
            key = "CO",
            name = "Carbon Monoxide (CO)",
            icon = "🚗",
            description = "A colorless, odorless, and tasteless poisonous gas formed by incomplete carbon combustion.",
            healthyRange = "0 - 4.4 ppm",
            dangerLevel = "> 9.4 ppm",
            epaStandard = "8-hr: 9 ppm | 1-hr: 35 ppm",
            sources = listOf(
                "Automobile engine exhaust gases",
                "Industrial furnaces, kilns, and boilers",
                "Residential heating equipment & gas appliances",
                "Forest fires and wood-burning stoves"
            ),
            healthEffects = listOf(
                "Reduces oxygen binding to hemoglobin in blood",
                "Causes chest pain in individuals with heart disease",
                "Dizziness, headaches, impaired vision, and fatigue",
                "High concentrations are fatal and toxic to central nervous system"
            )
        ),
        "O3" to PollutantDetailInfo(
            key = "O3",
            name = "Ground-Level Ozone (O3)",
            icon = "☀️",
            description = "A secondary pollutant gas created when NOx and volatile organic compounds (VOCs) react in strong sunlight.",
            healthyRange = "0 - 0.054 ppm",
            dangerLevel = "> 0.070 ppm",
            epaStandard = "8-hr: 70 ppb (0.070 ppm)",
            sources = listOf(
                "Photochemical reaction of traffic exhaust in sunlight",
                "Chemical solvent evaporation & paint vapors",
                "Industrial hydrocarbon emissions",
                "Gasoline vapor leakage"
            ),
            healthEffects = listOf(
                "Deep lung tissue irritation and sunburn-like respiratory damage",
                "Coughing, throat tightness, and painful deep breathing",
                "Aggravation of chronic bronchitis, emphysema, and asthma",
                "Permanent structural lung impairment after long exposure"
            )
        )
    )

    val aqiLevelsList = listOf(
        AqiScaleLevel(
            range = "0-50",
            level = "Good",
            colorHex = "#98EC85",
            textColorHex = "#000000",
            description = "Air quality is satisfactory, and air pollution poses little or no risk.",
            healthAdvice = "Ideal conditions for outdoor sports, exercise, and activities."
        ),
        AqiScaleLevel(
            range = "51-100",
            level = "Moderate",
            colorHex = "#FFC000",
            textColorHex = "#000000",
            description = "Air quality is acceptable. Unusually sensitive individuals may experience minor symptoms.",
            healthAdvice = "Unusually sensitive people should consider limiting prolonged outdoor exertion."
        ),
        AqiScaleLevel(
            range = "101-150",
            level = "Unhealthy to Sensitive Groups",
            colorHex = "#FF7E00",
            textColorHex = "#000000",
            description = "Members of sensitive groups (asthma, heart issues, elderly) may experience health effects.",
            healthAdvice = "Sensitive groups should limit prolonged outdoor exertion and wear masks."
        ),
        AqiScaleLevel(
            range = "151-200",
            level = "Unhealthy",
            colorHex = "#E51A1A",
            textColorHex = "#FFFFFF",
            description = "Everyone may begin to experience health effects; sensitive groups may experience serious effects.",
            healthAdvice = "Everyone should limit prolonged outdoor exertion. Close windows and use air purifiers."
        ),
        AqiScaleLevel(
            range = "201-300",
            level = "Very Unhealthy",
            colorHex = "#8F3F97",
            textColorHex = "#FFFFFF",
            description = "Health alert: The risk of health effects is significantly increased for the entire population.",
            healthAdvice = "Everyone should limit outdoor exertion. Keep indoor air clean with high-efficiency filters."
        ),
        AqiScaleLevel(
            range = "301-500",
            level = "Hazardous",
            colorHex = "#660014",
            textColorHex = "#FFFFFF",
            description = "Health warning of emergency conditions: entire population is likely to be affected.",
            healthAdvice = "Everyone should avoid outdoor exertion. Stay indoors with windows sealed."
        )
    )

    val sensitiveGroupsAdviceList = listOf(
        SensitiveGroupAdvice(
            title = "Asthma & Respiratory Patients",
            icon = "🫁",
            summary = "Inhaling fine particulates or ground-level ozone causes airway inflammation and triggers asthma attacks.",
            adviceList = listOf(
                "Keep quick-relief inhalers accessible at all times when AQI > 100",
                "Avoid outdoor workouts during peak afternoon sunlight when Ozone peaks",
                "Use HEPA air purifiers indoors during moderate to poor AQI days"
            )
        ),
        SensitiveGroupAdvice(
            title = "Heart & Cardiovascular Patients",
            icon = "❤️",
            summary = "PM2.5 enters the bloodstream, causing vascular inflammation, high blood pressure, and strain on the heart.",
            adviceList = listOf(
                "Monitor blood pressure closely on Unhealthy AQI days",
                "Reduce heavy physical exertion outdoors when AQI exceeds 100",
                "Seek immediate emergency medical care if experiencing chest discomfort"
            )
        ),
        SensitiveGroupAdvice(
            title = "Children & Elderly Adults",
            icon = "👶",
            summary = "Children breathe more air per pound of body weight, and seniors have reduced respiratory reserves.",
            adviceList = listOf(
                "Keep recess and physical playtime indoors when AQI > 150",
                "Ensure indoor environments stay ventilated with clean air filters",
                "Encourage hydration and wear N95 masks during outdoor transit"
            )
        ),
        SensitiveGroupAdvice(
            title = "Outdoor Workers & Athletes",
            icon = "🏃",
            summary = "Heavy respiration during prolonged exertion increases overall pollutant intake by up to 10 times.",
            adviceList = listOf(
                "Schedule strenuous physical activity early in the morning when AQI is lowest",
                "Take frequent breaks in air-conditioned or filtered indoor spaces",
                "Wear tight-fitting N95/KN95 respirators on polluted work sites"
            )
        )
    )

    val projectTeamMembers = listOf(
        mapOf(
            "name" to "Indrajit Shaw",
            "role" to "Team Leader & Backend Developer",
            "college" to "Seacom Engineering College",
            "specialization" to "Machine Learning & IoT",
            "contributions" to "ML Model Development, Android App, Backend API Design"
        ),
        mapOf(
            "name" to "Aritra Guchhait",
            "role" to "Dataset Handler & UI/UX Designer",
            "college" to "Seacom Engineering College",
            "specialization" to "Machine Learning & User Experience",
            "contributions" to "UI/UX Design, Weather Integration, Dataset Handling"
        ),
        mapOf(
            "name" to "Debanand Ghosh",
            "role" to "Project Structure Designer & Air Quality Analyst",
            "college" to "Seacom Engineering College",
            "specialization" to "System Architecture & Feature Engineering",
            "contributions" to "System Architecture, Software Testing, API Integration"
        )
    )
}
