import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Jeddah session
session_2024 = fastf1.get_session(2024, "Saudi Arabia", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector data by driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# 2025 Saudi Arabian GP qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "VER", "PIA", "RUS", "LEC", "ANT", "SAI", "HAM", "TSU",
        "GAS", "NOR", "ALB", "LAW", "ALO", "HAD", "BEA", "STR",
        "DOO", "HUL", "OCO", "BOR"
    ],
    "QualifyingTime (s)": [
        87.294,  # Max Verstappen – 1:27.294 (pole) :contentReference[oaicite:2]{index=2}
        87.304,  # Oscar Piastri – +0.010 :contentReference[oaicite:3]{index=3}
        87.407,  # George Russell – +0.113 :contentReference[oaicite:4]{index=4}
        87.670,  # Charles Leclerc – +0.376 :contentReference[oaicite:5]{index=5}
        87.866,  # Andrea Kimi Antonelli – +0.572 :contentReference[oaicite:6]{index=6}
        88.164,  # Carlos Sainz – +0.870 :contentReference[oaicite:7]{index=7}
        88.201,  # Lewis Hamilton – +0.907 :contentReference[oaicite:8]{index=8}
        88.204,  # Yuki Tsunoda – +0.910 :contentReference[oaicite:9]{index=9}
        88.367,  # Pierre Gasly – +1.073 :contentReference[oaicite:10]{index=10}
        None,    # Lando Norris – entró en Q3 pero no realizó tiempo (crash) :contentReference[oaicite:11]{index=11}
        88.109,  # Alexander Albon – Q2 1:28.109 (no Q3) :contentReference[oaicite:12]{index=12}
        88.191,  # Liam Lawson – Q2 1:28.191 :contentReference[oaicite:13]{index=13}
        88.303,  # Fernando Alonso – Q2 1:28.303 :contentReference[oaicite:14]{index=14}
        88.418,  # Isack Hadjar – Q2 1:28.418 :contentReference[oaicite:15]{index=15}
        88.648,  # Oliver Bearman – Q2 1:28.648 :contentReference[oaicite:16]{index=16}
        88.645,  # Lance Stroll – Q1 1:28.645 (no Q2/Q3) :contentReference[oaicite:17]{index=17}
        88.739,  # Jack Doohan – Q1 1:28.739 :contentReference[oaicite:18]{index=18}
        88.782,  # Nico Hülkenberg – Q1 1:28.782 :contentReference[oaicite:19]{index=19}
        89.092,  # Esteban Ocon – Q1 1:29.092 :contentReference[oaicite:20]{index=20}
        89.462   # Gabriel Bortoleto – Q1 1:29.462 :contentReference[oaicite:21]{index=21}
    ]
})
# Before Arabian GP
driver_points = {
    "PIA": 74, "RUS": 63, "NOR": 77, "VER": 69, "HAM": 25,
    "LEC": 32, "HAD": 4, "ANT": 30, "TSU": 5, "ALB": 18,
    "OCO": 14, "HUL": 6, "ALO": 0, "STR": 10, "SAI": 1,
    "GAS": 6, "BEA": 5, "DOO": 0, "BOR": 0, "LAW": 0
}
qualifying_2025["DriverPoints"] = qualifying_2025["Driver"].map(driver_points)


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


# Posiciones en Japan 2025 
japan_positions = { 
    "PIA": 3, "RUS": 5, "NOR": 2, "VER": 1, "HAM": 7,
    "LEC": 4, "HAD": 8, "ANT": 6, "TSU": 12, "ALB": 9,
    "OCO": 18, "HUL": 16, "ALO": 11, "STR": 20, "SAI": 14,
    "GAS": 13, "BEA": 10, "DOO": 15, "BOR": 19, "LAW": 17
}
qualifying_2025["Japan_Position"] = qualifying_2025["Driver"].map(japan_positions)

bahrain_positions = {
    "PIA": 1,  "RUS": 2,  "LEC": 4,  "ANT": 11,  "GAS": 7,
    "NOR": 3,  "VER": 6,  "SAI": 19,  "HAM": 5,  "TSU": 9,
    "DOO": 14, "HAD": 13, "ALO": 15, "OCO": 8, "ALB": 12,
    "LAW": 16, "BOR": 18, "STR": 17, "BEA": 10, "HUL": 20
}
qualifying_2025["Bahrain_Position"] = qualifying_2025["Driver"].map(bahrain_positions)


# wet driver performance from the script
driver_wet_performance = {
    "VER": 0.975196, "HAM": 0.976464, "LEC": 0.975862, "NOR": 0.978179, "ALO": 0.972655,
    "RUS": 0.968678, "SAI": 0.978754, "TSU": 0.996338, "OCO": 0.981810, "GAS": 0.978832, "STR": 0.979857
}
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)


API_KEY = "2934b77bde8ee3d18209b6c57c339c9c"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=21.4225&lon=39.1818&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()
forecast_time = "2025-04-20 18:00:00"
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else 20

# only take into account wet performance if chance is greater than 75% for rain
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]




merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
last_year_winner = "VER" 
merged_data["LastYearWinner"] = (merged_data["Driver"] == last_year_winner).astype(int)

merged_data["QualifyingTime"] = merged_data["QualifyingTime"] ** 2

# Define las características completas que quieres usar
features = [
    "QualifyingTime", "RainProbability", 
    "Temperature", "TotalSectorTime (s)",
    "DriverPoints", "TeamPoints",
    "Australia_Position", "China_Position", "Japan_Position", "Bahrain_Position"
]

# Obtiene los tiempos de vuelta medios (target)
mean_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
merged_data = merged_data.merge(mean_lap_times, on="Driver", how="left")

# SOLO para entrenamiento: filtrar pilotos con datos históricos
train_data = merged_data[~merged_data["LapTime (s)"].isna()]
X_train_full = train_data[features].fillna(0)
y_train_full = train_data["LapTime (s)"]

# Dividir en entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=39)
model.fit(X_train, y_train)

# Evalúa el modelo
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# PARA PREDICCIÓN: todos los pilotos (incluidos los nuevos)
X_pred = merged_data[features].fillna(0)
merged_data["PredictedRaceTime (s)"] = model.predict(X_pred)

# Ordenar por tiempo de carrera predicho
final_results = merged_data.sort_values(by="PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Saudi Arabian GP Winner 🏁\n")
print(final_results[["Driver", "Team", "PredictedRaceTime (s)"]])

# Plot feature importances
feature_importance = model.feature_importances_
features_names = X_pred.columns

plt.figure(figsize=(12, 8))
plt.barh(features_names, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.savefig('feature_importance_saudi.png')
plt.show()

# Check correlation between features and target
corr_matrix = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", 
    "LastYearWinner", "TotalSectorTime (s)", "LapTime (s)",
    "DriverPoints", "TeamPoints"
]].corr()

print(corr_matrix)
