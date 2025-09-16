import fastf1
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("f1_cache")

# load 2024 Bahrain race, lap time, sector times
session_2024 = fastf1.get_session(2024, "Bahrain", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# get average sector times
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# 2025 Bahrain GP quali data
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "PIA", "RUS", "LEC", "ANT", "GAS", "NOR", "VER", "SAI", "HAM", "TSU",
        "DOO", "HAD", "ALO", "OCO", "ALB", "LAW", "BOR", "STR", "BEA", "HUL"
    ],
    "QualifyingTime (s)": [
        89.841,  # Oscar Piastri – 1:29.841 :contentReference[oaicite:1]{index=1}
        90.009,  # George Russell – +0.168 → 1:30.009 :contentReference[oaicite:2]{index=2}
        90.175,  # Charles Leclerc – +0.334 → 1:30.175 :contentReference[oaicite:3]{index=3}
        90.213,  # Andrea Kimi Antonelli – +0.372 → 1:30.213 :contentReference[oaicite:4]{index=4}
        90.216,  # Pierre Gasly – +0.375 → 1:30.216 :contentReference[oaicite:5]{index=5}
        90.267,  # Lando Norris – +0.426 → 1:30.267 :contentReference[oaicite:6]{index=6}
        90.423,  # Max Verstappen – +0.582 → 1:30.423 :contentReference[oaicite:7]{index=7}
        90.680,  # Carlos Sainz – +0.839 → 1:30.680 :contentReference[oaicite:8]{index=8}
        90.772,  # Lewis Hamilton – +0.931 → 1:30.772 :contentReference[oaicite:9]{index=9}
        91.303,  # Yuki Tsunoda – +1.462 → 1:31.303 :contentReference[oaicite:10]{index=10}
        91.245,  # Jack Doohan – Q2 time +0.791 → 1:31.245 :contentReference[oaicite:11]{index=11}
        91.271,  # Isack Hadjar – Q2 +0.817 → 1:31.271 :contentReference[oaicite:12]{index=12}
        91.886,  # Fernando Alonso – Q2 +1.432 → 1:31.886 :contentReference[oaicite:13]{index=13}
        91.594,  # Esteban Ocon – Q1 1:31.594 :contentReference[oaicite:14]{index=14}
        92.040,  # Alexander Albon – Q1 1:32.040 :contentReference[oaicite:15]{index=15}
        92.165,  # Liam Lawson – Q1 1:32.165 :contentReference[oaicite:16]{index=16}
        92.186,  # Gabriel Bortoleto – Q1 1:32.186 :contentReference[oaicite:17]{index=17}
        92.283,  # Lance Stroll – Q1 1:32.283 :contentReference[oaicite:18]{index=18}
        92.376,   # Oliver Bearman – Q1 1:32.376 :contentReference[oaicite:19]{index=19}
        92.067   # Nico Hülkenberg – Q1 1:32.067 :contentReference[oaicite:20]{index=20}
    ]
})


# add wet performance factor
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

# Añadir datos de puntos del piloto antes del GP de Bahrein 2025
driver_points = {
    "PIA": 49, "RUS": 45, "NOR": 62, "VER": 61, "HAM": 15,
    "LEC": 20, "HAD": 4, "ANT": 30, "TSU": 3, "ALB": 18,
    "OCO": 10, "HUL": 6, "ALO": 0, "STR": 10, "SAI": 1,
    "GAS": 0, "BEA": 5, "DOO": 0, "BOR": 0, "LAW": 0
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


# Posiciones en Japan 2025 
japan_positions = { 
    "PIA": 3, "RUS": 5, "NOR": 2, "VER": 1, "HAM": 7,
    "LEC": 4, "HAD": 8, "ANT": 6, "TSU": 12, "ALB": 9,
    "OCO": 18, "HUL": 16, "ALO": 11, "STR": 20, "SAI": 14,
    "GAS": 13, "BEA": 10, "DOO": 15, "BOR": 19, "LAW": 17
}
qualifying_2025["Japan_Position"] = qualifying_2025["Driver"].map(japan_positions)

# weather data
API_KEY = ""
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=26.0325&lon=50.5106&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()

forecast_time = "2025-04-30 15:00:00"
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else 20

# merge sector time data
merged_data = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# target: average lap time from 2024 race (puede tener menos pilotos que qualifying_2025)
mean_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
merged_data = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")
merged_data = merged_data.merge(mean_lap_times, on="Driver", how="left")  # 'LapTime (s)' será NaN si no hay datos 2024

merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

features = [
    "QualifyingTime (s)", 
    "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", 
    "WetPerformanceFactor", "RainProbability", "Temperature",
    "DriverPoints", "TeamPoints",
    "Australia_Position", "China_Position","Japan_Position"
]

# SOLO para entrenamiento y validación: pilotos con datos históricos
train_data = merged_data[~merged_data["LapTime (s)"].isna()]
X_train_full = train_data[features].fillna(0)
y_train_full = train_data["LapTime (s)"]

X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=38)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Evalúa el modelo
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# PARA PREDICCIÓN: todos los pilotos (incluidos los nuevos)
X_pred = merged_data[features].fillna(0)
merged_data["PredictedRaceTime (s)"] = model.predict(X_pred)

merged_data = merged_data.sort_values(by="PredictedRaceTime (s)")

print("\n🏁 Predicted 2025 Bahrain GP Winner 🏁\n")
print(merged_data[["Driver", "PredictedRaceTime (s)"]])

# Plot feature importances
feature_importance = model.feature_importances_
features = X_pred.columns  # Usar X_pred en lugar de X que no está definido

plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()
