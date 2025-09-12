# 2025 F1 Telemetry Predictions

🏎️ **F1 Telemetry Predictions 2025 - Gradient Boosting Machine Learning Model**

Welcome to the F1 Telemetry Predictions 2025 repository! This project is inspired by the outstanding work of Mar Antaya and aims to deliver advanced Formula 1 race predictions using machine learning and telemetry data.

## 🚀 Project Overview
At the core of this project is a **Gradient Boosting Machine Learning model** that predicts race outcomes based on telemetry data, historical results, and qualifying sessions. The system leverages:
- FastF1 API for detailed race and telemetry data
- 2024 race results and 2025 qualifying session data
- Feature engineering techniques to enhance prediction accuracy

## 📊 Data Sources
- **FastF1 API:** Retrieves lap times, race results, and telemetry information
- **2025 Qualifying Data:** Used for generating updated predictions
- **Historical F1 Results:** Processed and used for model training

## 🏁 How It Works
1. **Data Collection:** The script pulls relevant F1 data using the FastF1 API.
2. **Preprocessing & Feature Engineering:** Lap times are converted, driver names normalized, and race data is structured for analysis.
3. **Model Training:** A **Gradient Boosting Regressor** is trained with 2024 race results.
4. **Prediction:** The model predicts race times and ranks drivers for the 2025 season.
5. **Evaluation:** Model performance is measured using **Mean Absolute Error (MAE)**.

### Dependencies
- `fastf1`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## File Structure
Each race prediction file is numbered according to the official calendar, e.g., `prediction1.py` for Australia, `prediction2.py` for China, etc.

## 🔧 Usage
To run a prediction script:
```bash
python3 prediction1.py
```
Expected output:
```
🏁 Predicted 2025 Australian GP Winner 🏁
Driver: Charles Leclerc, Predicted Race Time: 82.67s
...
🔍 Model Error (MAE): 3.22 seconds
```

## 📈 Model Performance
Mean Absolute Error (MAE) is used to evaluate the accuracy of the predictions; lower values indicate more precise results.

## 📌 Future Improvements
- Incorporate **weather conditions** as a feature
- Add **pit stop strategies** to the model
- Explore **deep learning** models for improved accuracy
- Latest predictions will be posted before each 2025 F1 race by @mar_antaya on Instagram and TikTok

---

🏎️ **Start predicting F1 races like a data scientist!** 🚀  
Inspired by the work of Mar Antaya.
