import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Japanese GP race session, lap and sector times
session_2024 = fastf1.get_session(2024, "Japan", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

qualifying_2025 = pd.DataFrame({
    "Driver": [
        "VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI",
        "HUL", "OCO", "STR", "ANT", "HAD", "BEA", "LAW", "BOR", "DOO", "ALB"
    ],
    "QualifyingTime (s)": [
        86.983,  # Max Verstappen
        86.995,  # Lando Norris
        87.027,  # Oscar Piastri
        87.299,  # Charles Leclerc
        87.318,  # George Russell
        87.610,  # Lewis Hamilton
        87.822,  # Pierre Gasly
        87.897,  # Fernando Alonso
        88.000,  # Yuki Tsunoda
        87.836,  # Carlos Sainz
        88.570,  # Nico Hülkenberg
        88.696,  # Esteban Ocon
        89.271,  # Lance Stroll
        87.555,  # Andrea Kimi Antonelli
        87.569,  # Isack Hadjar
        87.867,  # Oliver Bearman
        87.906,  # Liam Lawson
        88.622,  # Gabriel Bortoleto
        88.877,  # Jack Doohan
        87.615   # Alexander Albon
    ]
})

# Add wet performance factor from script
# Created using results from Canada 2024 and Canada 2023 same gp one with rain and one without
driver_wet_performance = {
    "VER": 0.975196, 
    "HAM": 0.976464,  
    "LEC": 0.975862,  
    "NOR": 0.978179,  
    "ALO": 0.972655,  
    "RUS": 0.968678,  
    "SAI": 0.978754,  
    "TSU": 0.996338,  
    "OCO": 0.981810,  
    "GAS": 0.978832,  
    "STR": 0.979857   
}
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

# Añadir datos de puntos del piloto antes del GP de Japón 2025
driver_points = {
    "PIA": 34, "RUS": 35, "NOR": 44, "VER": 36, "HAM": 9,
    "LEC": 8, "HAD": 0, "ANT": 22, "TSU": 3, "ALB": 16,
    "OCO": 10, "HUL": 6, "ALO": 0, "STR": 10, "SAI": 1,
    "GAS": 0, "BEA": 4, "DOO": 0, "BOR": 0, "LAW": 0
}
qualifying_2025["DriverPoints"] = qualifying_2025["Driver"].map(driver_points)

# Mapeo de equipos
driver_team_mapping = {
    "PIA": "McLaren", "NOR": "McLaren",
    "RUS": "Mercedes", "ANT": "Mercedes",
    "VER": "Red Bull", "TSU": "Red Bull",
    "LEC": "Ferrari", "HAM": "Ferrari",
    "LAW": "RB", "HAD": "RB",
    "ALB": "Williams", "SAI": "Williams",
    "DOO": "Alpine", "GAS": "Alpine",
    "OCO": "Haas", "BEA": "Haas",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "HUL": "Sauber", "BOR": "Sauber"
}
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_team_mapping)

# Puntos por equipo
team_points = {}
for team in set(driver_team_mapping.values()):
    team_drivers = [d for d, t in driver_team_mapping.items() if t == team]
    team_points[team] = sum(driver_points.get(d, 0) for d in team_drivers)
qualifying_2025["TeamPoints"] = qualifying_2025["Team"].map(team_points)

# Posiciones en Australia 2025
australia_positions = { 
    "PIA": 9, "RUS": 3, "NOR": 1, "VER": 2, "HAM": 10,
    "LEC": 8, "HAD": 20, "ANT": 4, "TSU": 12, "ALB": 5,
    "OCO": 13, "HUL": 7, "ALO": 17, "STR": 6, "SAI": 18,
    "GAS": 11, "BEA": 14, "DOO": 19, "BOR": 16, "LAW": 15
}
qualifying_2025["Australia_Position"] = qualifying_2025["Driver"].map(australia_positions)

# Posiciones en China 2025 
china_positions = { 
    "PIA": 1, "RUS": 3, "NOR": 2, "VER": 4, "HAM": 6,
    "LEC": 5, "HAD": 14, "ANT": 8, "TSU": 19, "ALB": 9,
    "OCO": 7, "HUL": 18, "ALO": 20, "STR": 12, "SAI": 13,
    "GAS": 11, "BEA": 10, "DOO": 15, "BOR": 17, "LAW": 16
}
qualifying_2025["China_Position"] = qualifying_2025["Driver"].map(china_positions)



# Weather Data
API_KEY = "2934b77bde8ee3d18209b6c57c339c9c"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=34.8823&lon=136.5845&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()
print(weather_data)

# Extract the relevant weather data for the race (Sunday at 2pm local time)
forecast_time = "2025-04-05 14:00:00"
forecast_data = None
for forecast in weather_data["list"]:
    if forecast["dt_txt"] == forecast_time:
        forecast_data = forecast
        break

if forecast_data:
    rain_probability = forecast_data["pop"]
    temperature = forecast_data["main"]["temp"]  
else:
    rain_probability = 0 
    temperature = 20 

# Merge qualifying data with sector times and mean lap times (left join para mantener todos los pilotos)
mean_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
merged_data = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")
merged_data = merged_data.merge(mean_lap_times, on="Driver", how="left")  # 'LapTime (s)' será NaN si no hay datos 2024

# Create weather features for the model
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Define feature set (Qualifying + Sector Times + Weather + Wet Performance Factor)
features = [
    "QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "WetPerformanceFactor", "RainProbability", "Temperature",
    "DriverPoints", "TeamPoints",
    "Australia_Position", "China_Position"
]

# Solo pilotos con datos históricos para entrenar y validar
train_data = merged_data[~merged_data["LapTime (s)"].isna()]
X_train_full = train_data[features].fillna(0)
y_train_full = train_data["LapTime (s)"]

# Split para validación
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=38)

# Entrena el modelo
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Evalúa el modelo
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Predice para todos los pilotos (incluidos los nuevos)
X_pred = merged_data[features].fillna(0)
merged_data["PredictedRaceTime (s)"] = model.predict(X_pred)

qualifying_2025 = merged_data

qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

print("\n🏁 Predicted 2025 Japanese GP Winner🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

