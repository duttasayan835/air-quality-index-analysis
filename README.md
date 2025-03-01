# Advanced Air Quality Index (AQI) Analysis

A comprehensive web application for analyzing air quality, predicting AQI levels, planning eco-friendly routes, and assessing health impacts.

## Features

### 🔍 AQI Prediction & Visualization
- Input air pollutant values and get AQI predictions
- 3D visualization of pollutant data
- Personalized health risk assessment
- Detailed recommendations based on AQI levels

### 🗺️ Green Route Planning
- Plan eco-friendly travel routes between locations
- View air quality along your route
- Compare different travel modes (walking, cycling, driving)
- Get travel recommendations based on AQI levels

### 📊 Health Impact Analysis
- Create a personal health profile
- Visualize health risks based on AQI levels
- 3D visualization of how pollutants affect different organs
- Receive personalized health recommendations

### 🏙️ Urban Planning Insights
- Analyze air quality in different city zones
- Get smart city recommendations
- View sustainability metrics

## Technologies Used

- **Frontend**: Streamlit, Plotly, Folium
- **Backend**: Python, XGBoost
- **APIs**: OpenWeather, OpenRoute, Carbon
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, Folium, Streamlit components

## Installation

1. Clone the repository:
```
git clone https://github.com/duttasayan835/air-quality-index-analysis.git
cd air-quality-index-analysis
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENROUTE_API_KEY=your_openroute_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
CARBON_API_KEY=your_carbon_api_key
```

4. Run the application:
```
streamlit run app.py
```

## Model Files

The application requires the following model files in the project directory:
- `best_xgb_model.pkl`: The trained XGBoost model
- `scaler.pkl`: The feature scaler
- `label_encoder_classes.npy`: The label encoder classes

## Usage

1. Navigate to the different tabs to access various features
2. In the AQI Prediction tab, enter pollutant values to get predictions
3. In the Green Route Planning tab, enter start and destination points
4. In the Health Impact Analysis tab, create your health profile
5. In the Urban Planning Insights tab, view city zone analysis

## Acknowledgments

- OpenWeather API for air quality data
- OpenRoute API for route planning
- Various open-source libraries and tools used in this project
