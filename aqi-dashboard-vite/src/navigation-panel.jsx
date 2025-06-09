"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Home,
  BookOpen,
  Users,
  AlertTriangle,
  Heart,
  Shield,
  Clock,
  Activity,
  Wind,
  Sun,
  Cloud,
  Zap,
  Factory,
  Leaf,
  Target,
  CheckCircle,
  XCircle,
  Star,
  GraduationCap,
  School,
  Award,
} from "lucide-react"
import { EnhancedAQIDashboard } from "./components/AQIDashboard"

export default function NavigationPanel({ isOpen, onToggle, currentSection, onSectionChange }) {
  const [activeTab, setActiveTab] = useState("overview")

  const pollutantData = {
    "PM2.5": {
      name: "Fine Particulate Matter (PM2.5)",
      icon: "🌫️",
      description: "Particles smaller than 2.5 micrometers in diameter",
      sources: ["Vehicle emissions", "Industrial processes", "Wildfires", "Power plants"],
      healthEffects: [
        "Respiratory problems and reduced lung function",
        "Cardiovascular disease and heart attacks",
        "Premature death in people with heart or lung disease",
        "Aggravated asthma and irregular heartbeat",
      ],
      epaStandards: {
        annual: "12 μg/m³",
        daily: "35 μg/m³",
      },
    },
    PM10: {
      name: "Coarse Particulate Matter (PM10)",
      icon: "💨",
      description: "Particles smaller than 10 micrometers in diameter",
      sources: ["Dust storms", "Construction sites", "Agricultural activities", "Unpaved roads"],
      healthEffects: [
        "Irritation of eyes, nose, and throat",
        "Coughing and difficulty breathing",
        "Aggravated asthma",
        "Reduced lung function",
      ],
      epaStandards: {
        daily: "150 μg/m³",
      },
    },
    NO2: {
      name: "Nitrogen Dioxide (NO2)",
      icon: "🏭",
      description: "A reddish-brown gas with a sharp, biting odor",
      sources: ["Vehicle emissions", "Power plants", "Industrial facilities", "Gas stoves"],
      healthEffects: [
        "Respiratory tract irritation",
        "Increased susceptibility to respiratory infections",
        "Aggravated asthma",
        "Formation of ground-level ozone",
      ],
      epaStandards: {
        annual: "53 ppb",
        hourly: "100 ppb",
      },
    },
    SO2: {
      name: "Sulfur Dioxide (SO2)",
      icon: "⚗️",
      description: "A colorless gas with a sharp, pungent smell",
      sources: ["Coal and oil combustion", "Metal smelting", "Petroleum refining", "Volcanoes"],
      healthEffects: [
        "Respiratory problems and breathing difficulties",
        "Aggravated asthma and chronic bronchitis",
        "Eye irritation",
        "Cardiovascular problems",
      ],
      epaStandards: {
        hourly: "75 ppb",
        daily: "140 ppb",
      },
    },
    CO: {
      name: "Carbon Monoxide (CO)",
      icon: "🚗",
      description: "A colorless, odorless, and tasteless gas",
      sources: ["Vehicle emissions", "Industrial processes", "Residential heating", "Wildfires"],
      healthEffects: [
        "Reduced oxygen delivery to organs and tissues",
        "Chest pain in people with heart disease",
        "Impaired vision and reduced brain function",
        "Death at high concentrations",
      ],
      epaStandards: {
        "8-hour": "9 ppm",
        "1-hour": "35 ppm",
      },
    },
    O3: {
      name: "Ground-level Ozone (O3)",
      icon: "☀️",
      description: "A gas formed when pollutants react in sunlight",
      sources: ["Vehicle emissions", "Industrial facilities", "Chemical solvents", "Gasoline vapors"],
      healthEffects: [
        "Respiratory tract irritation and inflammation",
        "Reduced lung function",
        "Aggravated asthma",
        "Increased susceptibility to respiratory infections",
      ],
      epaStandards: {
        "8-hour": "70 ppb",
      },
    },
  }

  const aqiLevels = [
    {
      range: "0-50",
      level: "Good",
      color: "#00E400",
      description: "Air quality is satisfactory, and air pollution poses little or no risk.",
      healthAdvice: "Enjoy outdoor activities!",
    },
    {
      range: "51-100",
      level: "Moderate",
      color: "#FFFF00",
      description:
        "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.",
      healthAdvice: "Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    },
    {
      range: "101-150",
      level: "Unhealthy for Sensitive Groups",
      color: "#FF7E00",
      description:
        "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
      healthAdvice: "Sensitive groups should limit prolonged outdoor exertion.",
    },
    {
      range: "151-200",
      level: "Unhealthy",
      color: "#FF0000",
      description:
        "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.",
      healthAdvice: "Everyone should limit prolonged outdoor exertion.",
    },
    {
      range: "201-300",
      level: "Very Unhealthy",
      color: "#8F3F97",
      description: "Health alert: The risk of health effects is increased for everyone.",
      healthAdvice: "Everyone should limit outdoor exertion.",
    },
    {
      range: "301-500",
      level: "Hazardous",
      color: "#7E0023",
      description: "Health warning of emergency conditions: everyone is more likely to be affected.",
      healthAdvice: "Everyone should avoid outdoor exertion.",
    },
  ]

  const teamMembers = [
    {
      name: "Indrajit Shaw",
      role: "Team Leader & Backend Developer",
      education: "B.Tech in Computer Science & Engineering",
      year: "Final Year",
      college: "Seacom Engineering College",
      specialization: "Machine Learning & IoT",
      contributions: ["ML Model Development", "Backend API Design", "Data Analysis"],
    },
    {
      name: "Aritra Guchhait",
      role: "Dataset Handeler & UI/UX Designer",
      education: "B.Tech in Computer Science & Engineering",
      year: "Final Year",
      college: "Seacom Engineering College",
      specialization: "Machine Learning & User Experience",
      contributions: ["Frontend Development", "UI/UX Design", "Weather Integration", "Dataset Handling"],
    },
    {
      name: "Debanand Ghosh",
      role: "Project Structure Designer & Air Quality Analyst",
      education: "B.Tech in Computer Science & Engineering",
      year: "Final Year",
      college: "Seacom Engineering College",
      specialization: "System Architecture & ML Model Feature Engineering",
      contributions: ["System Architecture", "Software Testing", "API Integration"],
    },
  ]

  const userGuideSteps = [
    {
      step: 1,
      title: "Check Weather Conditions",
      description: "View real-time weather data that affects air quality in your location.",
      icon: <Cloud className="w-6 h-6" />,
      details: [
        "The weather section automatically loads your location's current conditions",
        "Weather data includes temperature, humidity, wind speed, and atmospheric pressure",
        "You can refresh the data or use weather-based pollutant estimates",
      ],
    },
    {
      step: 2,
      title: "Enter Pollutant Values",
      description: "Input current pollutant concentrations manually or use weather estimates.",
      icon: <Wind className="w-6 h-6" />,
      details: [
        "Enter values for PM2.5, PM10, NO2, SO2, CO, and O3",
        "All fields must be filled before making predictions",
        "Use the 'Use Weather-Based Estimates' button for quick setup",
        "Values should be in the specified units (μg/m³, ppb, ppm)",
      ],
    },
    {
      step: 3,
      title: "Get Current AQI Prediction",
      description: "Generate instant AQI prediction with health recommendations.",
      icon: <Activity className="w-6 h-6" />,
      details: [
        "Click 'Predict Current AQI' to get immediate results",
        "View AQI value, level, and color-coded status",
        "Receive personalized health advice based on the AQI level",
        "See detailed pollutant breakdown and analysis charts",
      ],
    },
    {
      step: 4,
      title: "View 24-Hour Forecast",
      description: "Get hourly AQI predictions for the next 24 hours.",
      icon: <Clock className="w-6 h-6" />,
      details: [
        "Generate comprehensive 24-hour forecasts",
        "View statistical analysis (min, max, average, median)",
        "Get smart recommendations for outdoor activities",
        "See hourly breakdown with trend analysis",
      ],
    },
    {
      step: 5,
      title: "Check 7-Day Forecast",
      description: "Plan ahead with weekly air quality predictions.",
      icon: <Sun className="w-6 h-6" />,
      details: [
        "Generate 7-day air quality forecasts",
        "View weekly statistics and trends",
        "Identify best and worst days for outdoor activities",
        "Compare weekday vs weekend air quality patterns",
      ],
    },
    {
      step: 6,
      title: "Take Protective Actions",
      description: "Follow health recommendations based on AQI predictions.",
      icon: <Shield className="w-6 h-6" />,
      details: [
        "Follow color-coded health advisories",
        "Adjust outdoor activities based on AQI levels",
        "Use air purifiers during high pollution periods",
        "Wear masks when AQI exceeds unhealthy levels",
      ],
    },
  ]

  const precautions = {
    "Good (0-50)": [
      "Perfect time for outdoor activities and exercise",
      "Open windows for natural ventilation",
      "Ideal for outdoor sports and recreation",
    ],
    "Moderate (51-100)": [
      "Generally safe for outdoor activities",
      "Sensitive individuals should monitor symptoms",
      "Consider indoor activities if you're unusually sensitive",
    ],
    "Unhealthy for Sensitive Groups (101-150)": [
      "Limit prolonged outdoor exertion if you're sensitive",
      "Children, elderly, and people with respiratory conditions should be cautious",
      "Consider wearing masks during outdoor activities",
    ],
    "Unhealthy (151-200)": [
      "Everyone should limit prolonged outdoor exertion",
      "Wear N95 masks when going outside",
      "Keep windows closed and use air purifiers",
      "Avoid outdoor exercise and sports",
    ],
    "Very Unhealthy (201-300)": [
      "Limit all outdoor activities",
      "Stay indoors as much as possible",
      "Use high-quality air purifiers",
      "Seek medical attention if experiencing symptoms",
    ],
    "Hazardous (301-500)": [
      "Avoid all outdoor activities",
      "Stay indoors with air purification",
      "Seek immediate medical attention for any symptoms",
      "Consider evacuation if possible",
    ],
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Navigation Header */}
      <div className="bg-white shadow-lg border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center gap-3">
              <div className="text-2xl">🌍</div>
              <div>
                <h1 className="text-2xl font-bold text-gray-800">AQI Prediction System</h1>
                <p className="text-sm text-gray-600">Advanced Air Quality Monitoring & Forecasting</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <Badge variant="outline" className="hidden sm:flex">
                Real-time Monitoring
              </Badge>
              <Badge variant="outline" className="hidden sm:flex">
                ML-Powered Predictions
              </Badge>
            </div>
          </div>
        </div>
      </div>

      {/* Main Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <Tabs defaultValue="home" className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="home" className="flex items-center gap-2">
              <Home className="w-4 h-4" />
              Home
            </TabsTrigger>
            <TabsTrigger value="guide" className="flex items-center gap-2">
              <BookOpen className="w-4 h-4" />
              User Guide
            </TabsTrigger>
            <TabsTrigger value="aqi" className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              AQI Dashboard
            </TabsTrigger>
            <TabsTrigger value="about" className="flex items-center gap-2">
              <Users className="w-4 h-4" />
              About Us
            </TabsTrigger>
          </TabsList>

          {/* HOME SECTION */}
          <TabsContent value="home" className="space-y-8">
            {/* Introduction */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-2xl">
                  <Leaf className="w-6 h-6 text-green-600" />
                  Understanding Air Quality Index (AQI)
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold mb-3">What is AQI?</h3>
                    <p className="text-gray-700 leading-relaxed">
                      The Air Quality Index (AQI) is a standardized system used to communicate how polluted the air
                      currently is or how polluted it is forecast to become. It translates complex air quality data into
                      a simple numerical scale from 0 to 500, with corresponding colors and health messages.
                    </p>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Why is AQI Important?</h3>
                    <p className="text-gray-700 leading-relaxed">
                      AQI helps you understand what local air quality means to your health. It provides guidance on when
                      to limit outdoor activities and when air quality is good for outdoor exercise and activities. Our
                      system uses advanced machine learning to predict future AQI levels.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* AQI Levels */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5 text-blue-600" />
                  AQI Levels & Health Implications
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {aqiLevels.map((level, index) => (
                    <div
                      key={index}
                      className="border rounded-lg p-4"
                      style={{ borderLeftColor: level.color, borderLeftWidth: "4px" }}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <Badge
                            style={{ backgroundColor: level.color, color: level.color === "#FFFF00" ? "#000" : "#fff" }}
                          >
                            {level.range}
                          </Badge>
                          <h4 className="font-semibold">{level.level}</h4>
                        </div>
                        {level.range === "0-50" && <CheckCircle className="w-5 h-5 text-green-500" />}
                        {level.range.startsWith("151") && <AlertTriangle className="w-5 h-5 text-orange-500" />}
                        {level.range.startsWith("301") && <XCircle className="w-5 h-5 text-red-500" />}
                      </div>
                      <p className="text-gray-700 mb-2">{level.description}</p>
                      <p className="text-sm font-medium text-blue-700">{level.healthAdvice}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* US EPA Guidelines */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5 text-blue-600" />
                  US EPA Air Quality Standards
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-blue-800 mb-2">National Ambient Air Quality Standards (NAAQS)</h3>
                    <p className="text-blue-700 text-sm">
                      The EPA has established National Ambient Air Quality Standards for six principal pollutants, which
                      are called "criteria" pollutants. Our system monitors and predicts levels for all these key
                      pollutants.
                    </p>
                  </div>

                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-3">
                      <h4 className="font-semibold">Primary Standards</h4>
                      <p className="text-sm text-gray-600">
                        Set to protect public health, including the health of sensitive populations such as children,
                        elderly, and people with asthma.
                      </p>
                    </div>
                    <div className="space-y-3">
                      <h4 className="font-semibold">Secondary Standards</h4>
                      <p className="text-sm text-gray-600">
                        Set to protect public welfare, including protection against decreased visibility and damage to
                        animals, crops, vegetation, and buildings.
                      </p>
                    </div>
                  </div>

                  <div className="bg-yellow-50 p-4 rounded-lg">
                    <h4 className="font-semibold text-yellow-800 mb-2">EPA AQI Calculation Method</h4>
                    <p className="text-yellow-700 text-sm">
                      Our system follows EPA's official AQI calculation methodology, using the highest AQI value among
                      all monitored pollutants. The AQI is calculated using the formula that converts pollutant
                      concentrations to a standardized scale.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Pollutants Information */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Factory className="w-5 h-5 text-gray-600" />
                  Major Air Pollutants & Health Effects
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="overview" className="w-full">
                  <TabsList className="grid w-full grid-cols-4 mb-6">
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="particulates">Particulates</TabsTrigger>
                    <TabsTrigger value="gases">Gases</TabsTrigger>
                    <TabsTrigger value="standards">EPA Standards</TabsTrigger>
                  </TabsList>

                  <TabsContent value="overview" className="space-y-4">
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(pollutantData).map(([key, pollutant]) => (
                        <Card key={key} className="border-2 hover:shadow-md transition-shadow">
                          <CardContent className="pt-4">
                            <div className="text-center space-y-3">
                              <div className="text-3xl">{pollutant.icon}</div>
                              <h4 className="font-semibold">{key}</h4>
                              <p className="text-sm text-gray-600">{pollutant.description}</p>
                              <Badge variant="outline" className="text-xs">
                                {pollutant.sources.length} Major Sources
                              </Badge>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </TabsContent>

                  <TabsContent value="particulates" className="space-y-6">
                    {["PM2.5", "PM10"].map((key) => {
                      const pollutant = pollutantData[key]
                      return (
                        <Card key={key}>
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                              <span className="text-2xl">{pollutant.icon}</span>
                              {pollutant.name}
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <p className="text-gray-700">{pollutant.description}</p>
                            <div className="grid md:grid-cols-2 gap-4">
                              <div>
                                <h5 className="font-semibold mb-2 flex items-center gap-2">
                                  <Factory className="w-4 h-4" />
                                  Major Sources
                                </h5>
                                <ul className="space-y-1">
                                  {pollutant.sources.map((source, idx) => (
                                    <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                                      <div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
                                      {source}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                              <div>
                                <h5 className="font-semibold mb-2 flex items-center gap-2">
                                  <Heart className="w-4 h-4 text-red-500" />
                                  Health Effects
                                </h5>
                                <ul className="space-y-1">
                                  {pollutant.healthEffects.map((effect, idx) => (
                                    <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                                      <AlertTriangle className="w-3 h-3 text-orange-500 flex-shrink-0" />
                                      {effect}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </TabsContent>

                  <TabsContent value="gases" className="space-y-6">
                    {["NO2", "SO2", "CO", "O3"].map((key) => {
                      const pollutant = pollutantData[key]
                      return (
                        <Card key={key}>
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                              <span className="text-2xl">{pollutant.icon}</span>
                              {pollutant.name}
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <p className="text-gray-700">{pollutant.description}</p>
                            <div className="grid md:grid-cols-2 gap-4">
                              <div>
                                <h5 className="font-semibold mb-2 flex items-center gap-2">
                                  <Factory className="w-4 h-4" />
                                  Major Sources
                                </h5>
                                <ul className="space-y-1">
                                  {pollutant.sources.map((source, idx) => (
                                    <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                                      <div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
                                      {source}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                              <div>
                                <h5 className="font-semibold mb-2 flex items-center gap-2">
                                  <Heart className="w-4 h-4 text-red-500" />
                                  Health Effects
                                </h5>
                                <ul className="space-y-1">
                                  {pollutant.healthEffects.map((effect, idx) => (
                                    <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                                      <AlertTriangle className="w-3 h-3 text-orange-500 flex-shrink-0" />
                                      {effect}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </TabsContent>

                  <TabsContent value="standards" className="space-y-4">
                    <div className="bg-blue-50 p-4 rounded-lg mb-6">
                      <h3 className="font-semibold text-blue-800 mb-2">EPA National Ambient Air Quality Standards</h3>
                      <p className="text-blue-700 text-sm">
                        These standards represent the maximum allowable concentrations of pollutants in outdoor air to
                        protect public health and welfare.
                      </p>
                    </div>

                    <div className="grid gap-4">
                      {Object.entries(pollutantData).map(([key, pollutant]) => (
                        <Card key={key}>
                          <CardContent className="pt-4">
                            <div className="flex items-center gap-3 mb-3">
                              <span className="text-2xl">{pollutant.icon}</span>
                              <h4 className="font-semibold">{key}</h4>
                            </div>
                            <div className="grid md:grid-cols-2 gap-4">
                              {Object.entries(pollutant.epaStandards).map(([period, value]) => (
                                <div key={period} className="bg-gray-50 p-3 rounded">
                                  <div className="text-sm text-gray-600 capitalize">{period} Average</div>
                                  <div className="font-semibold text-lg">{value}</div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>

            {/* Health Precautions */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5 text-green-600" />
                  Health Precautions by AQI Level
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {Object.entries(precautions).map(([level, actions]) => {
                    const aqiLevel = aqiLevels.find((l) => l.level === level.split(" (")[0])
                    return (
                      <div
                        key={level}
                        className="border rounded-lg p-4"
                        style={{ borderLeftColor: aqiLevel?.color, borderLeftWidth: "4px" }}
                      >
                        <h4 className="font-semibold mb-3 flex items-center gap-2">
                          <Badge
                            style={{
                              backgroundColor: aqiLevel?.color,
                              color: aqiLevel?.color === "#FFFF00" ? "#000" : "#fff",
                            }}
                          >
                            {level}
                          </Badge>
                        </h4>
                        <ul className="space-y-2">
                          {actions.map((action, idx) => (
                            <li key={idx} className="text-sm text-gray-700 flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                              {action}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* USER GUIDE SECTION */}
          <TabsContent value="guide" className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-2xl">
                  <BookOpen className="w-6 h-6 text-blue-600" />
                  How to Use the AQI Prediction System
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 leading-relaxed mb-6">
                  Our AQI Prediction System uses advanced machine learning algorithms to provide accurate air quality
                  forecasts. Follow this step-by-step guide to get the most out of our platform.
                </p>
              </CardContent>
            </Card>

            {/* Step-by-step Guide */}
            <div className="space-y-6">
              {userGuideSteps.map((step, index) => (
                <Card key={index}>
                  <CardContent className="pt-6">
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0">
                        <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                          {step.icon}
                        </div>
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <Badge variant="outline">Step {step.step}</Badge>
                          <h3 className="text-lg font-semibold">{step.title}</h3>
                        </div>
                        <p className="text-gray-700 mb-4">{step.description}</p>
                        <ul className="space-y-2">
                          {step.details.map((detail, idx) => (
                            <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
                              {detail}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Tips and Best Practices */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Star className="w-5 h-5 text-yellow-500" />
                  Tips & Best Practices
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-3">For Accurate Predictions</h4>
                    <ul className="space-y-2">
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        Use current, accurate pollutant measurements
                      </li>
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        Update weather data regularly
                      </li>
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        Consider local environmental factors
                      </li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-3">Health & Safety</h4>
                    <ul className="space-y-2">
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <Heart className="w-4 h-4 text-red-500" />
                        Check AQI before outdoor activities
                      </li>
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <Heart className="w-4 h-4 text-red-500" />
                        Follow health recommendations
                      </li>
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <Heart className="w-4 h-4 text-red-500" />
                        Monitor sensitive group members
                      </li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* AQI DASHBOARD SECTION */}
          <TabsContent value="aqi" className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-2xl">
                  <Activity className="w-6 h-6 text-blue-600" />
                  AQI Prediction Dashboard
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 leading-relaxed mb-6">
                  Use our advanced machine learning-powered dashboard to predict Air Quality Index (AQI) values,
                  generate forecasts, and get personalized health recommendations based on current pollutant levels.
                </p>
              </CardContent>
            </Card>

            <EnhancedAQIDashboard />
          </TabsContent>

          {/* ABOUT US SECTION */}
          <TabsContent value="about" className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-2xl">
                  <Users className="w-6 h-6 text-purple-600" />
                  About Our Team
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center space-y-4 mb-8">
                  <div className="text-4xl">🎓</div>
                  <h2 className="text-xl font-semibold">Final Year B.Tech Students</h2>
                  <p className="text-gray-700 leading-relaxed max-w-3xl mx-auto">
                    We are a dedicated team of Computer Science & Engineering students from Seacom Engineering College,
                    passionate about leveraging technology to address environmental challenges. Our AQI Prediction
                    System represents the culmination of our academic journey and our commitment to creating solutions
                    that benefit society.
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Team Members */}
            <div className="grid md:grid-cols-1 lg:grid-cols-3 gap-6">
              {teamMembers.map((member, index) => (
                <Card key={index} className="border-2 hover:shadow-lg transition-shadow">
                  <CardHeader className="text-center">
                    <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mx-auto mb-4 flex items-center justify-center text-white text-2xl font-bold">
                      {member.name
                        .split(" ")
                        .map((n) => n[0])
                        .join("")}
                    </div>
                    <CardTitle className="text-lg">{member.name}</CardTitle>
                    <p className="text-sm text-gray-600">{member.role}</p>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-sm">
                        <GraduationCap className="w-4 h-4 text-blue-500" />
                        <span>{member.education}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <School className="w-4 h-4 text-green-500" />
                        <span>{member.college}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <Award className="w-4 h-4 text-purple-500" />
                        <span>{member.year}</span>
                      </div>
                    </div>

                    <Separator />

                    <div>
                      <h5 className="font-semibold text-sm mb-2">Specialization</h5>
                      <p className="text-xs text-gray-600">{member.specialization}</p>
                    </div>

                    <div>
                      <h5 className="font-semibold text-sm mb-2">Key Contributions</h5>
                      <ul className="space-y-1">
                        {member.contributions.map((contribution, idx) => (
                          <li key={idx} className="text-xs text-gray-600 flex items-center gap-2">
                            <div className="w-1 h-1 bg-blue-500 rounded-full"></div>
                            {contribution}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Project Information */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-yellow-500" />
                  Project Overview
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-3">Technical Stack</h4>
                    <ul className="space-y-2">
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        React.js for Frontend
                      </li>
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        Python Flask for Backend API
                      </li>
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                        Machine Learning Models
                      </li>
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                        Weather API Integration
                      </li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-3">Key Features</h4>
                    <ul className="space-y-2">
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        Real-time AQI Prediction
                      </li>
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        24-hour & 7-day Forecasting
                      </li>
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        Weather Integration
                      </li>
                      <li className="text-sm text-gray-700 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        Health Recommendations
                      </li>
                    </ul>
                  </div>
                </div>

                <Separator />

                <div className="text-center space-y-4">
                  <h4 className="font-semibold">Academic Institution</h4>
                  <div className="bg-blue-50 p-6 rounded-lg">
                    <h3 className="text-xl font-bold text-blue-800 mb-2">Seacom Engineering College</h3>
                    <p className="text-blue-700">Department of Computer Science & Engineering</p>
                    <p className="text-sm text-blue-600 mt-2">Final Year Project - Academic Year 2024-25</p>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-green-50 to-blue-50 p-6 rounded-lg">
                  <h4 className="font-semibold text-center mb-3">Our Mission</h4>
                  <p className="text-gray-700 text-center leading-relaxed">
                    To develop an intelligent, accessible, and accurate air quality prediction system that empowers
                    individuals and communities to make informed decisions about their health and outdoor activities. We
                    believe that technology can play a crucial role in environmental awareness and public health
                    protection.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
