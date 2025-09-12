import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load FastF1 2024 Australian GP race session
session_2024 = fastf1.get_session(2024, "China", "R")
session_2024.load()

# Extract lap times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# 2025 Qualifying Data Chinese GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
               "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
               "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
    "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                           91.021, 91.079, 91.103, 91.638, 91.706,
                           91.625, 91.632, 91.688, 91.773, 91.840,
                           91.992, 92.018, 92.092, 92.141, 92.174]
})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Añadir datos de puntos del piloto antes del GP de China 2025
driver_points = {
    "PIA": 2, "RUS": 15, "NOR": 25, "VER": 18, "HAM": 1,
    "LEC": 4, "HAD": 0, "ANT": 12, "TSU": 0, "ALB": 10,
    "OCO": 0, "HUL": 6, "ALO": 0, "STR": 8, "SAI": 0,
    "GAS": 0, "BEA": 0, "DOO": 0, "BOR": 0, "LAW": 0
}

qualifying_2025["DriverPoints"] = qualifying_2025["DriverCode"].map(driver_points)

# Añadir mapeo de pilotos a equipos
driver_team_mapping = {
    "PIA": "McLaren", "NOR": "McLaren",
    "RUS": "Mercedes", "ANT": "Mercedes",
    "VER": "Red Bull", "LAW": "Red Bull",
    "LEC": "Ferrari", "HAM": "Ferrari",
    "TSU": "RB", "HAD": "RB",
    "ALB": "Williams", "SAI": "Williams",
    "DOO": "Alpine", "GAS": "Alpine",
    "OCO": "Haas", "BEA": "Haas",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "HUL": "Sauber", "BOR": "Sauber"
}

qualifying_2025["Team"] = qualifying_2025["DriverCode"].map(driver_team_mapping)

# Calcular puntos por equipo (suma de puntos de ambos pilotos)
team_points = {}
for team in set(driver_team_mapping.values()):
    team_drivers = [driver for driver, t in driver_team_mapping.items() if t == team]
    team_points[team] = sum(driver_points.get(driver, 0) for driver in team_drivers)

qualifying_2025["TeamPoints"] = qualifying_2025["Team"].map(team_points)



previous_race_positions = {
    # Posiciones del GP de Australia 2025 
    "PIA": 9, "RUS": 3, "NOR": 1, "VER": 2, "HAM": 10,
    "LEC": 8, "HAD": 20, "ANT": 4, "TSU": 12, "ALB": 5,
    "OCO": 13, "HUL": 7, "ALO": 17, "STR": 6, "SAI": 18,
    "GAS": 11, "BEA": 14, "DOO": 19, "BOR": 16, "LAW": 15
}

qualifying_2025["Australia_Position"] = qualifying_2025["DriverCode"].map(previous_race_positions)




# Merge 2025 Qualifying Data with 2024 Race Data
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

# Usar características ampliadas: tiempo de clasificación, puntos del piloto y puntos del equipo
X = merged_data[["QualifyingTime (s)", "DriverPoints", "TeamPoints","Australia_Position"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times and points data
X_pred = qualifying_2025[["QualifyingTime (s)", "DriverPoints", "TeamPoints","Australia_Position"]]
predicted_lap_times = model.predict(X_pred)
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Chinese GP Winner with Driver & Team Points 🏁\n")
print(qualifying_2025[["Driver", "Team", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

