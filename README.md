# 2025 F1 Telemetry Predictions

## Project Overview
At the core of this project is a **Gradient Boosting Machine Learning model** that predicts race outcomes based on telemetry data, historical results, and qualifying sessions. The system leverages:
- FastF1 API for detailed race and telemetry data
- 2024 race results and 2025 qualifying session data
- Feature engineering techniques to enhance prediction accuracy

## Data Sources
- **FastF1 API:** Retrieves lap times, race results, and telemetry information
- **2025 Qualifying Data:** Used for generating updated predictions
- **Historical F1 Results:** Processed and used for model training

## How It Works
1. **Data Collection:** The script pulls relevant F1 data using the FastF1 API.
2. **Preprocessing & Feature Engineering:** Lap times are converted, driver names normalized, and race data is structured for analysis.
3. **Model Training:** A **Gradient Boosting Regressor** is trained with 2024 race results.
4. **Prediction:** The model predicts race times and ranks drivers for the 2025 season.
5. **Evaluation:** Model performance is measured using **Mean Absolute Error (MAE)**.

### Dependencies
- `fastf1`
- `pandas`
- `scikit-learn`
- `matplotlib`

## File Structure
Each race prediction file is numbered according to the official calendar, e.g., `prediction1.py` for Australia, `prediction2.py` for China, etc.



Inspired by the work of Mar Antaya.
