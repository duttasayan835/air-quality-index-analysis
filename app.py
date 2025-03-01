import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
import datetime
from datetime import timedelta
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import json
from PIL import Image
import pydeck as pdk
import base64
from io import BytesIO
import os
from dotenv import load_dotenv
import time
import folium
from streamlit_folium import folium_static
import altair as alt
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, List, Tuple, Optional, Union
import sys
import streamlit.runtime.scriptrunner.script_runner as script_runner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set warning level for Streamlit loggers
logging.getLogger('streamlit').setLevel(logging.WARNING)
logging.getLogger('streamlit.runtime').setLevel(logging.WARNING)

# Create logger instance for this application
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define features list globally
features = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene"
]

# Initialize Streamlit configuration
if not st.runtime.exists():
    try:
        st.set_page_config(
            page_title="Air Quality Index Analysis",
            page_icon="🌍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass  # Page config already set

# Custom exception for model loading


class ModelLoadingError(Exception):
    pass

# Enhanced model loading with better error handling


@st.cache_resource
def load_ml_models() -> Tuple[object, object, np.ndarray]:
    """Load machine learning models with enhanced error handling and caching."""
    try:
        if not os.path.exists("best_xgb_model.pkl"):
            raise ModelLoadingError(
                "Model file 'best_xgb_model.pkl' not found")
        if not os.path.exists("scaler.pkl"):
            raise ModelLoadingError("Scaler file 'scaler.pkl' not found")
        if not os.path.exists("label_encoder_classes.npy"):
            raise ModelLoadingError(
                "Label encoder file 'label_encoder_classes.npy' not found")

        model = joblib.load("best_xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder_classes = np.load(
            "label_encoder_classes.npy", allow_pickle=True)

        logger.info("Successfully loaded ML models and preprocessing objects")
        return model, scaler, label_encoder_classes
    except ModelLoadingError as e:
        logger.error(f"Model loading error: {str(e)}")
        st.error(
            f"⚠️ {
                str(e)}. Please ensure all model files are present in the application directory.")
        return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error loading models: {str(e)}")
        st.error(
            "⚠️ Failed to load ML models. Please check the application logs for details.")
        return None, None, None


def initialize_session_state():
    """Initialize or update session state variables."""
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'health_conditions': [],
            'age_group': 'Adult (18-60)',
            'activity_level': 'Moderate',
            'sensitivity': 'Normal',
            'outdoor_duration': 2,
            'condition_severity': {}
        }
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None


# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    initialize_session_state()

# Load ML models
model, scaler, label_encoder_classes = load_ml_models()
st.session_state.model_loaded = model is not None

# API Keys from environment variables
OPENROUTE_API_KEY = os.environ.get("OPENROUTE_API_KEY", "")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
CARBON_API_KEY = os.environ.get("CARBON_API_KEY", "")

# Check if API keys are available
if not OPENROUTE_API_KEY or not OPENWEATHER_API_KEY or not CARBON_API_KEY:
    # Try to load from .env file directly if environment variables are not set
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)  # Force reload to ensure we get the latest values
        
        OPENROUTE_API_KEY = os.environ.get("OPENROUTE_API_KEY", "")
        OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
        CARBON_API_KEY = os.environ.get("CARBON_API_KEY", "")
        
        logger.info("Loaded API keys from .env file")
    except Exception as e:
        logger.error(f"Error loading API keys from .env file: {str(e)}")

# Debug: Print the first few characters of each API key
logger.info(f"OPENROUTE_API_KEY starts with: {OPENROUTE_API_KEY[:10] if OPENROUTE_API_KEY else 'empty'}")
logger.info(f"OPENWEATHER_API_KEY starts with: {OPENWEATHER_API_KEY[:10] if OPENWEATHER_API_KEY else 'empty'}")
logger.info(f"CARBON_API_KEY starts with: {CARBON_API_KEY[:10] if CARBON_API_KEY else 'empty'}")

# Display warning if keys are still missing
if not OPENROUTE_API_KEY or not OPENWEATHER_API_KEY or not CARBON_API_KEY:
    st.warning("""
    ⚠️ API keys are missing. Please set the following environment variables:
    - OPENROUTE_API_KEY
    - OPENWEATHER_API_KEY
    - CARBON_API_KEY

    Create a .env file in the project directory with these variables.
    """)
    
    # Log the API key status for debugging
    logger.warning(f"OPENROUTE_API_KEY present: {bool(OPENROUTE_API_KEY)}")
    logger.warning(f"OPENWEATHER_API_KEY present: {bool(OPENWEATHER_API_KEY)}")
    logger.warning(f"CARBON_API_KEY present: {bool(CARBON_API_KEY)}")
else:
    logger.info("All required API keys are loaded")

# Function to get coordinates from location name


def get_coordinates(location_name):
    try:
        geolocator = Nominatim(user_agent="air_quality_app")
        location = geolocator.geocode(location_name)
        if location:
            return [location.latitude, location.longitude]
        return None
    except GeocoderTimedOut:
        st.error(
            f"Error: Geocoding service timed out for {location_name}. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error: Could not geocode {location_name}. Error: {str(e)}")
        return None

# Function to get route details from OpenRoute service


def get_route(start_coords, end_coords, mode):
    try:
        # Check if API key is available
        if not OPENROUTE_API_KEY:
            st.error("Error: OpenRoute API key is missing. Please check your .env file.")
            logger.error("OpenRoute API key is missing")
            return None
            
        transport_mode = {
            "Walking": "foot-walking",
            "Cycling": "cycling-regular",
            "Driving": "driving-car"
        }

        # Log the API request for debugging
        logger.info(f"Requesting route from {start_coords} to {end_coords} using mode: {mode}")
        
        # Debug: Print the API key being used (temporary)
        logger.info(f"Using OpenRoute API key: {OPENROUTE_API_KEY}")
        
        # Make sure the API key is properly formatted (no quotes or whitespace)
        clean_api_key = OPENROUTE_API_KEY.strip().replace('"', '').replace("'", "")

        headers = {
            'Authorization': clean_api_key
        }

        body = {
            "coordinates": [
                start_coords[::-1],  # OpenRoute expects [lon, lat]
                end_coords[::-1]
            ],
            "instructions": "true",
            "geometry": "true",  # Ensure we get the full route geometry
            "format": "geojson"  # Request GeoJSON format for easier processing
        }

        response = requests.post(
            f'https://api.openrouteservice.org/v2/directions/{transport_mode[mode]}/json',
            json=body,
            headers=headers,
            timeout=10  # Add timeout to prevent hanging requests
        )

        # Check for specific error status codes
        if response.status_code == 401:
            st.error("Error: Invalid OpenRoute API key. Please check your .env file.")
            logger.error(f"Invalid OpenRoute API key. Response: {response.text}")
            return None
        elif response.status_code == 429:
            st.error("Error: Rate limit exceeded for OpenRoute API.")
            logger.error(f"Rate limit exceeded for OpenRoute API. Response: {response.text}")
            return None

        response.raise_for_status()  # Raise exception for other 4XX/5XX responses
        
        # Log successful response
        logger.info("Successfully retrieved route data")
        
        # Parse the response
        route_data = response.json()
        
        # Log the structure of the response to help with debugging
        logger.info(f"Route data keys: {route_data.keys()}")
        if 'routes' in route_data and len(route_data['routes']) > 0:
            logger.info(f"First route keys: {route_data['routes'][0].keys()}")
            if 'geometry' in route_data['routes'][0]:
                logger.info("Route contains geometry data")
                
        return route_data
    except requests.exceptions.Timeout:
        st.error("Error: The route planning service timed out. Please try again.")
        logger.error("OpenRoute API request timed out")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Error: HTTP error occurred while getting route: {e}")
        logger.error(f"HTTP error in OpenRoute API request: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error: An unexpected error occurred while getting route: {str(e)}")
        logger.error(f"Unexpected error in OpenRoute API request: {str(e)}")
        return None

# Function to get AQI data from OpenWeather


def get_aqi_data(lat, lon):
    try:
        # Check if API key is available
        if not OPENWEATHER_API_KEY:
            st.error("Error: OpenWeather API key is missing. Please check your .env file.")
            logger.error("OpenWeather API key is missing")
            return None
            
        # Log the API request for debugging
        logger.info(f"Requesting AQI data for coordinates: {lat}, {lon}")
        
        # Debug: Print the full API key being used (temporary)
        logger.info(f"Using OpenWeather API key: {OPENWEATHER_API_KEY}")
        
        # Make sure the API key is properly formatted (no quotes or whitespace)
        clean_api_key = OPENWEATHER_API_KEY.strip().replace('"', '').replace("'", "")
        
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={clean_api_key}"
        logger.info(f"Request URL: {url}")
        
        response = requests.get(url, timeout=10)  # Add timeout
        
        # Check for specific error status codes
        if response.status_code == 401:
            st.error("Error: Invalid OpenWeather API key. Please check your .env file.")
            logger.error(f"Invalid OpenWeather API key. Response: {response.text}")
            return None
        elif response.status_code == 429:
            st.error("Error: Rate limit exceeded for OpenWeather API.")
            logger.error(f"Rate limit exceeded for OpenWeather API. Response: {response.text}")
            return None
        
        response.raise_for_status()  # Raise exception for other 4XX/5XX responses
        
        # Log successful response
        logger.info("Successfully retrieved AQI data")
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Error: The air quality service timed out. Please try again.")
        logger.error("OpenWeather API request timed out")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Error: HTTP error occurred while getting AQI data: {e}")
        logger.error(f"HTTP error in OpenWeather API request: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error: An unexpected error occurred while getting AQI data: {str(e)}")
        logger.error(f"Unexpected error in OpenWeather API request: {str(e)}")
        return None

# Load the saved model and preprocessing objects


def load_model():
    global model, scaler, label_encoder_classes, model_loaded
    try:
        model = joblib.load("best_xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder_classes = np.load(
            "label_encoder_classes.npy", allow_pickle=True)
        model_loaded = True
        return True
    except BaseException:
        model_loaded = False
        return False


# Custom CSS for a more modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .info-text {
        font-size: 1rem;
        color: #4B5563;
        line-height: 1.5;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
    }
</style>
""", unsafe_allow_html=True)

# Create a title with custom styling
st.markdown(
    '<div class="main-header">🌍 Advanced Air Quality Index (AQI) Analysis</div>',
    unsafe_allow_html=True)

# Create tabs for different features
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 AQI Prediction & Visualization",
    "🗺️ Green Route Planning",
    "📊 Health Impact Analysis",
    "🏙️ Urban Planning Insights"
])

# Function to calculate health risk score


def calculate_health_risk_score(aqi_level, health_profile):
    base_risk = {
        'Good': 1,
        'Satisfactory': 2,
        'Moderate': 3,
        'Poor': 4,
        'Very Poor': 5,
        'Severe': 6
    }

    # Age factor
    age_weights = {
        "0-12": 1.4,
        "13-17": 1.2,
        "18-30": 1.0,
        "31-50": 1.1,
        "51-70": 1.3,
        "70+": 1.5
    }

    # Activity level factor
    activity_weights = {
        "Sedentary": 0.8,
        "Light": 0.9,
        "Moderate": 1.0,
        "Active": 1.2,
        "Very Active": 1.4
    }

    # Calculate condition severity score
    severity_weights = {"Mild": 1.2, "Moderate": 1.5, "Severe": 2.0}
    condition_score = 1.0
    for condition in health_profile['health_conditions']:
        if condition in health_profile.get('condition_severity', {}):
            severity = health_profile['condition_severity'][condition]
            condition_score = max(
                condition_score, severity_weights.get(
                    severity, 1.0))

    # Calculate final risk score
    risk_score = (
        base_risk.get(aqi_level, 3) *
        age_weights.get(health_profile['age_group'], 1.0) *
        activity_weights.get(health_profile['activity_level'], 1.0) *
        condition_score *
        # Outdoor exposure factor with default of 2 hours
        (1 + (health_profile.get('outdoor_duration', 2) / 24) * 0.5)
    )

    return min(risk_score, 10)  # Cap at 10

# Function to get recommendations


def get_recommendations(aqi_level, pollutant_levels):
    recommendations = {
        'Good': {
            'immediate_actions': [
                "✅ Enjoy outdoor activities",
                "🌳 Perfect time for exercise",
                "🏃‍♂️ Ideal for sports and recreation"
            ],
            'preventive_measures': [
                "📊 Monitor AQI regularly",
                "🌱 Maintain indoor plants",
                "♻️ Practice regular recycling"
            ],
            'long_term_strategies': [
                "🚲 Consider eco-friendly transportation",
                "🌿 Support local green initiatives",
                "💡 Use energy-efficient appliances"
            ]
        },
        'Satisfactory': {
            'immediate_actions': [
                "😷 Sensitive individuals may want to carry masks",
                "🏃‍♂️ Reduce prolonged outdoor exercise",
                "🚗 Consider carpooling"
            ],
            'preventive_measures': [
                "🏠 Keep windows closed during peak pollution hours",
                "🌿 Use air-purifying plants indoors",
                "💨 Check ventilation systems"
            ],
            'long_term_strategies': [
                "🔧 Regular vehicle maintenance",
                "♻️ Implement waste segregation",
                "🌱 Participate in local tree planting"
            ]
        },
        'Moderate': {
            'immediate_actions': [
                "😷 Wear N95 masks outdoors",
                "⚠️ Limit outdoor activities",
                "💧 Stay hydrated",
                "🏠 Use air purifiers indoors"
            ],
            'preventive_measures': [
                "🔍 Monitor symptoms of respiratory issues",
                "🚰 Install water sprinklers in gardens",
                "🌿 Create green barriers around home"
            ],
            'long_term_strategies': [
                "🏭 Support emission control policies",
                "🔋 Switch to renewable energy",
                "📱 Use AQI monitoring apps"
            ]
        },
        'Poor': {
            'immediate_actions': [
                "⚠️ Avoid outdoor activities",
                "😷 Use N95/N99 masks mandatory",
                "💊 Keep emergency medication handy",
                "🏥 Watch for health symptoms"
            ],
            'preventive_measures': [
                "🔒 Seal windows and doors",
                "💨 Use HEPA air purifiers",
                "🌡️ Monitor indoor air quality",
                "💦 Use air humidifiers"
            ],
            'long_term_strategies': [
                "🏗️ Install air filtration systems",
                "🌿 Create indoor clean air zones",
                "📞 Join local environmental groups"
            ]
        },
        'Very Poor': {
            'immediate_actions': [
                "⛔ Stay indoors",
                "🏥 Keep emergency contacts ready",
                "💊 Regular health monitoring",
                "😷 Use professional-grade masks"
            ],
            'preventive_measures': [
                "🏥 Create emergency health plan",
                "💨 Multiple air purifiers",
                "🔬 Regular health check-ups",
                "📱 Enable emergency alerts"
            ],
            'long_term_strategies': [
                "🏠 Install whole-house air filtration",
                "🚗 Switch to electric vehicles",
                "📢 Advocate for clean air policies"
            ]
        },
        'Severe': {
            'immediate_actions': [
                "🚨 Activate emergency protocols",
                "⛔ Complete indoor confinement",
                "🏥 Medical attention on standby",
                "💊 Follow medical advice strictly"
            ],
            'preventive_measures': [
                "🏥 Create safe rooms with purifiers",
                "📞 Emergency response plan",
                "🔬 Continuous health monitoring",
                "💉 Regular medical consultations"
            ],
            'long_term_strategies': [
                "🏗️ Relocate to cleaner areas",
                "📢 Community emergency planning",
                "🔬 Support air quality research"
            ]
        }
    }

    # Get pollutant-specific recommendations
    pollutant_specific = []
    if pollutant_levels["PM2.5"] > 60 or pollutant_levels["PM10"] > 100:
        pollutant_specific.append(
            "🔍 High particulate matter: Use HEPA filters")
    if pollutant_levels["NO2"] > 40 or pollutant_levels["NOx"] > 40:
        pollutant_specific.append(
            "🏭 High nitrogen oxides: Avoid high-traffic areas")
    if pollutant_levels["O3"] > 50:
        pollutant_specific.append(
            "☀️ High ozone: Avoid outdoors during peak sun hours")
    if pollutant_levels["CO"] > 4:
        pollutant_specific.append(
            "⚠️ High carbon monoxide: Check ventilation systems")

    return recommendations.get(aqi_level, {}), pollutant_specific

# Function to load Lottie animations


def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except BaseException:
        return None

# Function to set background based on AQI level


def set_background_by_aqi(aqi_level):
    # Dictionary mapping AQI levels to background images/animations
    aqi_backgrounds = {
        "Good": {
            "color": "#8BC34A",
            "lottie_url": "https://lottie.host/6e1b2bfd-c4d0-43cf-9e0a-b8d0c4e7c5b0/YQnKX6Xdoi.json",
            "description": "Clean, fresh air with minimal pollution. Perfect for outdoor activities."
        },
        "Satisfactory": {
            "color": "#CDDC39",
            "lottie_url": "https://lottie.host/dd7e9b44-9c53-4a7c-a3a3-d4e3c24c4d4c/fkTfZfwGAi.json",
            "description": "Air quality is acceptable, though there may be moderate health concerns for a small number of sensitive individuals."
        },
        "Moderate": {
            "color": "#FFC107",
            "lottie_url": "https://lottie.host/e8a24e2f-78b7-4384-a6f3-97d333ffa144/sPNcPXANYe.json",
            "description": "Air quality is acceptable; however, there may be some pollutants present that could affect sensitive groups."
        },
        "Poor": {
            "color": "#FF9800",
            "lottie_url": "https://lottie.host/9d3da5e9-d3df-4d3d-b6c8-db49fce7c1c1/HdmvTHBY5K.json",
            "description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
        },
        "Very Poor": {
            "color": "#F44336",
            "lottie_url": "https://lottie.host/f0ddc6d0-99a5-4fe3-ac5f-e1e1a8074142/lfYP0YfZlg.json",
            "description": "Health alert: everyone may begin to experience health effects. Members of sensitive groups may experience more serious health effects."
        },
        "Severe": {
            "color": "#9C27B0",
            "lottie_url": "https://lottie.host/3e2ca516-c320-44a8-8873-5a9c3f41a7e1/Pj1X6A7GYx.json",
            "description": "Health warnings of emergency conditions. The entire population is more likely to be affected."
        }
    }

    # Get background info for the given AQI level
    bg_info = aqi_backgrounds.get(aqi_level, aqi_backgrounds["Moderate"])

    # Set CSS for gradient background
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(to bottom, {bg_info["color"]}33, #f0f2f6);
        }}
        .aqi-card {{
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        .aqi-title {{
            color: {bg_info["color"]};
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .aqi-description {{
            font-size: 16px;
            color: #555;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    return bg_info

# Function to create 3D AQI visualization


def create_3d_aqi_visualization(input_data, aqi_level):
    # Create a 3D scatter plot for pollutants
    features = list(input_data.keys())
    values = list(input_data.values())

    # Normalize values for better visualization
    max_val = max(values)
    if max_val > 0:
        norm_values = [v / max_val * 100 for v in values]
    else:
        norm_values = values

    # Create color scale based on AQI level
    aqi_colors = {
        "Good": "#8BC34A",
        "Satisfactory": "#CDDC39",
        "Moderate": "#FFC107",
        "Poor": "#FF9800",
        "Very Poor": "#F44336",
        "Severe": "#9C27B0"
    }

    color = aqi_colors.get(aqi_level, "#FFC107")

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=[i for i in range(len(features))],
        y=[i % 3 for i in range(len(features))],  # Arrange in a grid
        z=norm_values,
        mode='markers',
        marker=dict(
            size=norm_values,
            color=norm_values,
            colorscale=[[0, 'green'], [0.5, 'yellow'], [1, color]],
            opacity=0.8,
            symbol='circle'
        ),
        text=features,
        hovertemplate='<b>%{text}</b><br>Value: %{marker.size:.1f}<extra></extra>'
    )])

    # Update layout
    fig.update_layout(
        title=f"3D Visualization of Air Pollutants - {aqi_level} AQI",
        scene=dict(
            xaxis=dict(title='', showticklabels=False),
            yaxis=dict(title='', showticklabels=False),
            zaxis=dict(title='Normalized Value')
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=500
    )

    return fig


def process_route_planning(start_point, destination, selected_mode):
    """Process route planning with proper error handling."""
    try:
        # Get coordinates
        start_coords = get_coordinates(start_point)
        end_coords = get_coordinates(destination)

        if not start_coords or not end_coords:
            return False, "Could not find coordinates for one or both locations. Please check the addresses."

        # Get route details
        route_data = get_route(start_coords, end_coords, selected_mode)
        if not route_data:
            return False, "Could not calculate route. Please try different locations or travel mode."

        # Process route data
        route = route_data["routes"][0]
        distance_km = route["summary"]["distance"] / 1000
        duration_min = route["summary"]["duration"] / 60

        # Get AQI data
        start_aqi_data = get_aqi_data(start_coords[0], start_coords[1])
        end_aqi_data = get_aqi_data(end_coords[0], end_coords[1])

        if not start_aqi_data or not end_aqi_data:
            return False, "Could not retrieve air quality data for the locations."

        return True, {
            "route": route,
            "distance_km": distance_km,
            "duration_min": duration_min,
            "start_aqi_data": start_aqi_data,
            "end_aqi_data": end_aqi_data,
            "start_point": start_point,
            "destination": destination,
            "start_coords": start_coords,
            "end_coords": end_coords
        }
    except Exception as e:
        return False, f"An error occurred: {str(e)}"


def display_route_info(result):
    """Display route information including distance, duration, and AQI data."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗺️ Route Details")
    
    # Display basic route information
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Distance", f"{result['distance_km']:.1f} km")
    with col2:
        st.metric("Duration", f"{result['duration_min']:.0f} min")

    # Create a map with the route
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Extract route coordinates
    coordinates = []
    
    # Debug the route data structure
    logger.info(f"Route data keys: {result['route'].keys()}")
    
    # Try to extract coordinates from the route geometry
    try:
        # Check if we have GeoJSON format geometry
        if 'geometry' in result['route']:
            geometry = result['route']['geometry']
            logger.info(f"Geometry type: {type(geometry)}")
            
            # Handle GeoJSON geometry object
            if isinstance(geometry, dict) and 'coordinates' in geometry:
                coordinates = geometry['coordinates']
                logger.info(f"Found {len(coordinates)} coordinates in GeoJSON geometry")
            
            # Handle encoded polyline string
            elif isinstance(geometry, str):
                try:
                    # Try to decode as GeoJSON
                    import json
                    geom_obj = json.loads(geometry)
                    if 'coordinates' in geom_obj:
                        coordinates = geom_obj['coordinates']
                        logger.info(f"Found {len(coordinates)} coordinates in decoded geometry string")
                except json.JSONDecodeError:
                    logger.warning("Geometry is not a valid JSON string")
                    
                    # Try to decode as polyline
                    try:
                        from polyline import decode
                        decoded = decode(geometry)
                        if decoded:
                            # Polyline format is [lat, lon] but we need [lon, lat]
                            coordinates = [[point[1], point[0]] for point in decoded]
                            logger.info(f"Found {len(coordinates)} coordinates in decoded polyline")
                    except Exception as e:
                        logger.error(f"Failed to decode polyline: {str(e)}")
        
        # If no coordinates yet, try to extract from segments
        if not coordinates and 'segments' in result['route']:
            logger.info("Extracting coordinates from segments")
            for segment in result['route']['segments']:
                # Try to get geometry from the segment
                if 'geometry' in segment:
                    if isinstance(segment['geometry'], dict) and 'coordinates' in segment['geometry']:
                        coordinates = segment['geometry']['coordinates']
                        logger.info(f"Found {len(coordinates)} coordinates in segment geometry")
                        break
                    elif isinstance(segment['geometry'], str):
                        try:
                            import json
                            geom = json.loads(segment['geometry'])
                            if 'coordinates' in geom:
                                coordinates = geom['coordinates']
                                logger.info(f"Found {len(coordinates)} coordinates in segment geometry string")
                                break
                        except Exception:
                            pass
                
                # If no geometry in segment, try to get from steps
                if not coordinates and 'steps' in segment:
                    logger.info(f"Extracting coordinates from {len(segment['steps'])} steps")
                    for step in segment['steps']:
                        if 'geometry' in step:
                            if isinstance(step['geometry'], dict) and 'coordinates' in step['geometry']:
                                coordinates.extend(step['geometry']['coordinates'])
                            elif isinstance(step['geometry'], str):
                                try:
                                    import json
                                    geom = json.loads(step['geometry'])
                                    if 'coordinates' in geom:
                                        coordinates.extend(geom['coordinates'])
                                except Exception:
                                    pass
    except Exception as e:
        logger.error(f"Error extracting route coordinates: {str(e)}")
    
    logger.info(f"Total coordinates extracted: {len(coordinates)}")
    
    if coordinates:
        # Center the map on the route
        center_lat = sum(coord[1] for coord in coordinates) / len(coordinates)
        center_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
        m.location = [center_lat, center_lon]
        m.zoom_start = 12
        
        # Add the route line
        folium.PolyLine(
            locations=[[coord[1], coord[0]] for coord in coordinates],
            weight=5,
            color='blue',
            opacity=0.8
        ).add_to(m)
        
        # Add markers for start and end points
        folium.Marker(
            [coordinates[0][1], coordinates[0][0]],
            popup='Start',
            icon=folium.Icon(color='green', icon='info-sign')
        ).add_to(m)
        folium.Marker(
            [coordinates[-1][1], coordinates[-1][0]],
            popup='End',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    else:
        # If no coordinates found, center map on the start/end coordinates we already have
        logger.warning("No route coordinates found to display on map")
        
        # Use the coordinates we already have from the result
        try:
            start_coords = result['start_coords']
            end_coords = result['end_coords']
            
            if start_coords and end_coords:
                center_lat = (start_coords[0] + end_coords[0]) / 2
                center_lon = (start_coords[1] + end_coords[1]) / 2
                m.location = [center_lat, center_lon]
                m.zoom_start = 10
                
                # Add markers for start and end points
                folium.Marker(
                    start_coords,
                    popup='Start',
                    icon=folium.Icon(color='green', icon='info-sign')
                ).add_to(m)
                folium.Marker(
                    end_coords,
                    popup='End',
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
                
                # Add a straight line between start and end
                folium.PolyLine(
                    locations=[start_coords, end_coords],
                    weight=3,
                    color='red',
                    opacity=0.6,
                    dash_array='5'
                ).add_to(m)
                
                logger.info(f"Added fallback route line from {start_coords} to {end_coords}")
            else:
                logger.warning("No start/end coordinates available for fallback map")
        except Exception as e:
            logger.error(f"Error setting default map view: {str(e)}")
    
    # Display the map
    folium_static(m)
    
    # Display AQI information
    st.subheader("🌬️ Air Quality Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Starting Point AQI**")
        if 'list' in result['start_aqi_data']:
            aqi = result['start_aqi_data']['list'][0]['main']['aqi']
            st.metric("AQI Level", get_aqi_category(aqi))
            st.write("Pollutant Levels:")
            for pollutant, value in result['start_aqi_data']['list'][0]['components'].items():
                st.write(f"- {pollutant}: {value:.1f} μg/m³")
    
    with col2:
        st.markdown("**Destination AQI**")
        if 'list' in result['end_aqi_data']:
            aqi = result['end_aqi_data']['list'][0]['main']['aqi']
            st.metric("AQI Level", get_aqi_category(aqi))
            st.write("Pollutant Levels:")
            for pollutant, value in result['end_aqi_data']['list'][0]['components'].items():
                st.write(f"- {pollutant}: {value:.1f} μg/m³")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_travel_recommendations(result, selected_mode):
    """Display travel recommendations based on route and AQI data."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💡 Travel Recommendations")
    
    # Get AQI levels
    start_aqi = result['start_aqi_data']['list'][0]['main']['aqi']
    end_aqi = result['end_aqi_data']['list'][0]['main']['aqi']
    start_aqi_category = get_aqi_category(start_aqi)
    end_aqi_category = get_aqi_category(end_aqi)
    
    # Generate recommendations based on mode and AQI
    recommendations = []
    
    # Mode-specific recommendations
    if selected_mode == "Walking":
        if max(start_aqi, end_aqi) >= 4:  # Poor or worse
            recommendations.append("⚠️ Consider wearing an N95 mask for protection")
            recommendations.append("🕒 Try to schedule your walk during less polluted hours")
        recommendations.append("💧 Stay hydrated during your walk")
        recommendations.append("🏃 Take breaks if needed, especially in high pollution areas")
        
    elif selected_mode == "Cycling":
        if max(start_aqi, end_aqi) >= 4:
            recommendations.append("⚠️ Wear appropriate respiratory protection")
            recommendations.append("🔄 Consider alternative transportation")
        recommendations.append("😷 Consider wearing a sports mask")
        recommendations.append("🚴 Choose less congested routes when possible")
        
    elif selected_mode == "Driving":
        recommendations.append("🚗 Keep windows closed in high pollution areas")
        recommendations.append("🌪️ Use recirculated air conditioning")
        if max(start_aqi, end_aqi) >= 4:
            recommendations.append("🔧 Ensure your vehicle's air filter is clean")
    
    # General recommendations based on AQI levels
    if max(start_aqi, end_aqi) >= 5:  # Very Poor or Severe
        recommendations.append("⚠️ Consider postponing non-essential travel")
        recommendations.append("🏥 Keep emergency contact information handy")
    
    # Display recommendations
    for rec in recommendations:
        st.write(rec)
    
    # Display carbon footprint estimate
    st.subheader("🌱 Environmental Impact")
    carbon_footprint = calculate_carbon_footprint(result['distance_km'], selected_mode)
    st.metric("Estimated Carbon Footprint", f"{carbon_footprint:.2f} kg CO2")
    
    st.markdown('</div>', unsafe_allow_html=True)

def get_aqi_category(aqi_value):
    """Convert AQI value to category."""
    if aqi_value == 1:
        return "Good"
    elif aqi_value == 2:
        return "Fair"
    elif aqi_value == 3:
        return "Moderate"
    elif aqi_value == 4:
        return "Poor"
    elif aqi_value == 5:
        return "Very Poor"
    else:
        return "Severe"

def calculate_carbon_footprint(distance_km, mode):
    """Calculate carbon footprint based on distance and mode of transport."""
    # CO2 emissions in kg per km (approximate values)
    emissions_per_km = {
        "Walking": 0,
        "Cycling": 0,
        "Driving": 0.2  # Average car emissions
    }
    return distance_km * emissions_per_km.get(mode, 0)

with tab1:
    st.markdown('<div class="sub-header">🔍 Air Quality Prediction & 3D Visualization</div>', unsafe_allow_html=True)
    
    # Introduction text
    st.markdown(
        """
        <div class="info-text">
        This interactive tool allows you to input air pollutant values and get an AQI prediction along with 
        <span class="highlight">stunning 3D visualizations</span> that change based on air quality levels.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    # Input fields for features
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Enter Pollutant Values")
    
    input_data = {}
    for i, feature in enumerate(features):
        if i < len(features) // 2:
            with col1:
                # Add validation for numeric inputs
                from utils.input_validation import validate_numeric_input
                value = st.number_input(
                    feature,
                    value=0.0,
                    help=f"Enter the {feature} value"
                )
                # Validate the input
                is_valid, validated_value = validate_numeric_input(
                    value, 
                    field_name=feature,
                    min_val=0.0,  # Assuming all air quality values should be non-negative
                    max_val=1000.0,  # Set a reasonable maximum value
                    error_location=col1
                )
                if is_valid:
                    input_data[feature] = validated_value
                else:
                    input_data[feature] = 0.0  # Default to safe value if invalid
        else:
            with col2:
                # Add validation for numeric inputs
                from utils.input_validation import validate_numeric_input
                value = st.number_input(
                    feature,
                    value=0.0,
                    help=f"Enter the {feature} value"
                )
                # Validate the input
                is_valid, validated_value = validate_numeric_input(
                    value, 
                    field_name=feature,
                    min_val=0.0,  # Assuming all air quality values should be non-negative
                    max_val=1000.0,  # Set a reasonable maximum value
                    error_location=col2
                )
                if is_valid:
                    input_data[feature] = validated_value
                else:
                    input_data[feature] = 0.0  # Default to safe value if invalid
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add predict button
    if st.button("Classify AQI", use_container_width=True):
        if st.session_state.model_loaded:
            # Create DataFrame from input data
            input_df = pd.DataFrame([input_data])
            
            # Scale the input features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            aqi_level = label_encoder_classes[prediction[0]]
            
            # Calculate health risk score
            risk_score = calculate_health_risk_score(aqi_level, st.session_state.user_profile)
            
            # Set background based on AQI level
            bg_info = set_background_by_aqi(aqi_level)
            
            # Display AQI card with Lottie animation
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Load and display Lottie animation
                lottie_json = load_lottieurl(bg_info["lottie_url"])
                if lottie_json:
                    st_lottie(lottie_json, height=200, key="aqi_animation")
            
            with col2:
                st.markdown(f'<div class="aqi-title">AQI Level: {aqi_level}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="aqi-description">{bg_info["description"]}</div>', unsafe_allow_html=True)
                
                # Display personalized alert if risk is high
                if risk_score > 5:
                    st.warning(f"⚠️ **Personal Health Alert**: Based on your health conditions, current AQI levels pose elevated risks. Take necessary precautions!")
                
                # Display health risk score
                st.metric("Personal Health Risk Score", f"{risk_score:.1f}/10")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display 3D AQI visualization
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔮 3D Pollutant Visualization")
            fig = create_3d_aqi_visualization(input_data, aqi_level)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Get recommendations
            general_recs, pollutant_recs = get_recommendations(aqi_level, input_data)
            
            # Display recommendations
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🌿 Recommendations")
            
            # Display immediate actions
            st.markdown("**Immediate Actions:**")
            for rec in general_recs.get('immediate_actions', []):
                st.write(rec)
                
            # Display preventive measures
            st.markdown("**Preventive Measures:**")
            for rec in general_recs.get('preventive_measures', []):
                st.write(rec)
                
            # Display long-term strategies
            st.markdown("**Long-term Strategies:**")
            for rec in general_recs.get('long_term_strategies', []):
                st.write(rec)
                
            # Display pollutant-specific recommendations
            if pollutant_recs:
                st.markdown("**Pollutant-Specific Recommendations:**")
                for rec in pollutant_recs:
                    st.write(rec)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Store the last prediction in session state
            st.session_state.last_prediction = {
                'input_data': input_data,
                'aqi_level': aqi_level,
                'risk_score': risk_score
            }
        else:
            st.error("⚠️ Model not loaded. Please check if model files are present in the application directory.")

with tab2:
    st.markdown(
        '<div class="sub-header">🗺️ Green Route Planning</div>',
        unsafe_allow_html=True)

    # Introduction text
    st.markdown(
        """
        <div class="info-text">
        Plan your travel routes with environmental consciousness. This tool helps you find the most eco-friendly
        routes between locations while considering air quality along the way.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Enhanced route planning inputs
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        start_point = st.text_input("Starting Point", "New Delhi, India")
        # Validate starting point
        if start_point:
            from utils.input_validation import validate_text_input, sanitize_input
            start_point = sanitize_input(start_point)
            if not validate_text_input(start_point, field_name="Starting Point",
                                       min_length=3, max_length=100,
                                       pattern=r'^[A-Za-z0-9\s\.,\-\']+$',
                                       error_location=col1):
                start_point = ""

    with col2:
        destination = st.text_input("Destination", "Gurgaon, India")
        # Validate destination
        if destination:
            from utils.input_validation import validate_text_input, sanitize_input
            destination = sanitize_input(destination)
            if not validate_text_input(destination, field_name="Destination",
                                       min_length=3, max_length=100,
                                       pattern=r'^[A-Za-z0-9\s\.,\-\']+$',
                                       error_location=col2):
                destination = ""

    st.markdown('</div>', unsafe_allow_html=True)

    # Travel mode selection with icons
    travel_modes = {
        "Walking 🚶": "Walking",
        "Cycling 🚲": "Cycling",
        "Driving 🚗": "Driving"
    }

    selected_mode_key = st.radio(
        "Travel Mode",
        options=list(travel_modes.keys()),
        horizontal=True
    )
    selected_mode = travel_modes[selected_mode_key]

    if st.button("Find Green Routes", use_container_width=True):
        if not start_point or not destination:
            st.warning("Please enter both starting point and destination.")
        else:
            with st.spinner("Calculating route and environmental metrics..."):
                success, result = process_route_planning(
                    start_point, destination, selected_mode)
                if not success:
                    st.error(result)
                else:
                    # Display route information
                    display_route_info(result)

                    # Display recommendations
                    display_travel_recommendations(result, selected_mode)

with tab3:
    st.markdown(
        '<div class="sub-header">📊 Health Impact Analysis</div>',
        unsafe_allow_html=True)

    # Introduction text
    st.markdown(
        """
        <div class="info-text">
        Understand how air quality affects your health based on your personal profile. This tool provides
        personalized health risk analysis and visualizes potential impacts of different pollutants.
        </div>
        """,
        unsafe_allow_html=True
    )

    # User Profile Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👤 Personal Health Profile")

    # Create columns for profile inputs
    col1, col2 = st.columns(2)

    with col1:
        # Age group selection - validate the selection against predefined
        # options
        age_options = [
            "Child (0-12)",
            "Teen (13-17)",
            "Adult (18-60)",
            "Senior (60+)"]
        age_index = age_options.index(
            st.session_state.user_profile['age_group']) if st.session_state.user_profile['age_group'] in age_options else 2
        age_group = st.selectbox(
            "Age Group:",
            age_options,
            index=age_index
        )
        st.session_state.user_profile['age_group'] = age_group

        # Activity level
        activity_level = st.select_slider(
            "Activity Level:",
            options=[
                "Sedentary",
                "Light",
                "Moderate",
                "Active",
                "Very Active"],
            value=st.session_state.user_profile['activity_level']
        )
        st.session_state.user_profile['activity_level'] = activity_level

    with col2:
        # Health sensitivity - validate the selection against predefined
        # options
        sensitivity_options = ["Low", "Normal", "High", "Very High"]
        sensitivity_index = sensitivity_options.index(
            st.session_state.user_profile.get(
                'sensitivity', 'Normal')) if st.session_state.user_profile.get(
            'sensitivity', 'Normal') in sensitivity_options else 1
        sensitivity = st.selectbox(
            "Sensitivity to Air Pollution:",
            sensitivity_options,
            index=sensitivity_index
        )
        st.session_state.user_profile['sensitivity'] = sensitivity

        # Health conditions - validate the selection against predefined options
        health_condition_options = [
            "Asthma",
            "COPD",
            "Heart Disease",
            "Allergies",
            "Pregnancy",
            "Diabetes",
            "Immunocompromised"]
        # Filter out any user-provided health conditions that aren't in our
        # predefined list
        valid_health_conditions = [
            condition for condition in st.session_state.user_profile['health_conditions'] if condition in health_condition_options]

        health_conditions = st.multiselect(
            "Health Conditions:",
            health_condition_options,
            default=valid_health_conditions
        )
        st.session_state.user_profile['health_conditions'] = health_conditions

    st.markdown('</div>', unsafe_allow_html=True)

    # Health Risk Analysis Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Health Risk Analysis")

    # Create sample data for different AQI levels
    aqi_levels = [
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor",
        "Very Poor",
        "Severe"]

    # Calculate risk scores for each AQI level
    risk_scores = [
        calculate_health_risk_score(
            level,
            st.session_state.user_profile) for level in aqi_levels]

    # Create risk data for table
    risk_data = {
        "AQI Level": aqi_levels,
        "Your Risk Score (0-10)": [f"{score:.1f}" for score in risk_scores]
    }

    # Create a color scale for risk scores
    risk_colors = []
    for score in risk_scores:
        if score < 3:
            color = "#8BC34A"  # Green
        elif score < 5:
            color = "#CDDC39"  # Light green-yellow
        elif score < 7:
            color = "#FFC107"  # Yellow
        elif score < 8.5:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
        risk_colors.append(color)

    # Create a 3D bar chart for risk visualization
    fig = go.Figure(data=[
        go.Bar(
            x=aqi_levels,
            y=risk_scores,
            marker_color=risk_colors,
            text=[f"{score:.1f}" for score in risk_scores],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Your Health Risk by AQI Level",
        xaxis_title="AQI Level",
        yaxis_title="Risk Score (0-10)",
        yaxis_range=[0, 10],
        height=400,
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Display risk interpretation
    st.subheader("📋 Risk Interpretation")

    # Create columns for risk levels
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div style="background-color: #8BC34A33; padding: 10px; border-radius: 5px;">
            <h4 style="color: #8BC34A;">Low Risk (0-3)</h4>
            <p>Minimal health concerns. Most people can continue normal activities.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="background-color: #FFC10733; padding: 10px; border-radius: 5px;">
            <h4 style="color: #FFC107;">Moderate Risk (3-7)</h4>
            <p>Some health effects possible. Consider reducing prolonged outdoor activities.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="background-color: #F4433633; padding: 10px; border-radius: 5px;">
            <h4 style="color: #F44336;">High Risk (7-10)</h4>
            <p>Serious health effects possible. Avoid outdoor activities and use air purifiers.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # 3D Health Impact Visualization Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🫁 3D Health Impact Visualization")

    # Create tabs for different visualizations
    viz_tab1, viz_tab2 = st.tabs(["Organ Impact", "Pollutant Sensitivity"])

    with viz_tab1:
        # Create 3D visualization of how pollutants affect different organs
        organs = ["Lungs", "Heart", "Brain", "Skin", "Liver", "Kidneys"]

        # Different impact levels based on health conditions
        has_respiratory = any(
            cond in [
                "Asthma",
                "COPD",
                "Allergies"] for cond in health_conditions)
        has_cardiovascular = any(
            cond in ["Heart Disease"] for cond in health_conditions)
        has_other = any(
            cond in [
                "Diabetes",
                "Immunocompromised",
                "Pregnancy"] for cond in health_conditions)

        # Base impact values
        impact_values = {
            "Lungs": 60,
            "Heart": 40,
            "Brain": 30,
            "Skin": 20,
            "Liver": 15,
            "Kidneys": 10
        }

        # Adjust based on conditions
        if has_respiratory:
            impact_values["Lungs"] += 30
            impact_values["Heart"] += 10

        if has_cardiovascular:
            impact_values["Heart"] += 30
            impact_values["Brain"] += 15

        if has_other:
            for organ in organs:
                impact_values[organ] += 10

        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=[i for i in range(len(organs))],
            y=[i % 3 for i in range(len(organs))],
            z=list(impact_values.values()),
            mode='markers+text',
            marker=dict(
                size=list(impact_values.values()),
                color=list(impact_values.values()),
                colorscale='Viridis',
                opacity=0.8,
                symbol='circle'
            ),
            text=organs,
            hovertemplate='<b>%{text}</b><br>Impact: %{marker.size}<extra></extra>'
        )])

        # Update layout
        fig.update_layout(
            title="Potential Impact of Air Pollution on Different Organs",
            scene=dict(
                xaxis=dict(title='', showticklabels=False),
                yaxis=dict(title='', showticklabels=False),
                zaxis=dict(title='Impact Level')
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    with viz_tab2:
        # Create 3D visualization of sensitivity to different pollutants
        pollutants = ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"]

        # Base sensitivity values
        sensitivity_values = {
            "PM2.5": 80,
            "PM10": 70,
            "NO2": 60,
            "O3": 50,
            "SO2": 40,
            "CO": 30
        }

        # Adjust based on conditions and sensitivity
        sensitivity_multiplier = {
            "Low": 0.7,
            "Normal": 1.0,
            "High": 1.3,
            "Very High": 1.6
        }

        # Apply multiplier
        for pollutant in pollutants:
            sensitivity_values[pollutant] *= sensitivity_multiplier[sensitivity]

        # Additional adjustments based on health conditions
        if "Asthma" in health_conditions:
            sensitivity_values["PM2.5"] *= 1.4
            sensitivity_values["O3"] *= 1.3

        if "COPD" in health_conditions:
            sensitivity_values["PM2.5"] *= 1.5
            sensitivity_values["NO2"] *= 1.4

        if "Heart Disease" in health_conditions:
            sensitivity_values["PM2.5"] *= 1.3
            sensitivity_values["CO"] *= 1.4

        if "Allergies" in health_conditions:
            sensitivity_values["PM2.5"] *= 1.2
            sensitivity_values["O3"] *= 1.2

        # Create 3D visualization data
        x = np.array([i for i in range(len(pollutants))])
        y = np.array([0 for _ in range(len(pollutants))])
        z = np.zeros(len(pollutants))
        
        fig = go.Figure(data=[go.Mesh3d(
            x=x,
            y=y,
            z=z,
            intensity=list(sensitivity_values.values()),
            text=pollutants,
            hovertemplate='<b>%{text}</b><br>Sensitivity: %{intensity:.1f}<extra></extra>',
            colorscale=[[0, '#8BC34A'], [0.5, '#FFC107'], [1, '#F44336']],
            intensitymode='cell',
            cmin=min(sensitivity_values.values()),
            cmax=max(sensitivity_values.values()),
            showscale=True
        )])

        # Update layout
        fig.update_layout(
            title=f"Your Sensitivity to Different Pollutants (Based on Health Profile)",
            scene=dict(
                xaxis=dict(title='', ticktext=pollutants, tickvals=list(range(len(pollutants)))),
                yaxis=dict(title=''),
                zaxis=dict(title='Sensitivity Level')
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Recommendations Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💡 Personalized Health Recommendations")

    # Generate personalized recommendations based on health profile
    recommendations = []

    # Base recommendations for everyone
    recommendations.append("Monitor local air quality regularly and plan activities accordingly")
    recommendations.append("Stay hydrated to help your body process and eliminate toxins")

    # Age-specific recommendations
    if age_group == "Child (0-12)":
        recommendations.append("Limit outdoor playtime during high pollution days")
        recommendations.append("Ensure children's schools have good indoor air quality")
    elif age_group == "Teen (13-17)":
        recommendations.append("Be cautious with outdoor sports during poor air quality days")
        recommendations.append("Use masks during outdoor activities in polluted areas")
    elif age_group == "Adult (18-60)":
        recommendations.append("Consider air quality when planning outdoor workouts")
        recommendations.append("Use air purifiers in home and workplace")
    else:  # Senior
        recommendations.append("Stay indoors during peak pollution hours")
        recommendations.append("Consult with healthcare provider about additional precautions")

    # Condition-specific recommendations
    if "Asthma" in health_conditions:
        recommendations.append("Keep rescue inhalers accessible at all times")
        recommendations.append("Consider wearing an N95 mask during moderate to poor air quality days")

    if "COPD" in health_conditions:
        recommendations.append("Follow your doctor's medication plan strictly during poor air quality days")
        recommendations.append("Consider using supplemental oxygen as prescribed during high pollution events")

    if "Heart Disease" in health_conditions:
        recommendations.append("Monitor your heart rate and blood pressure during poor air quality days")
        recommendations.append("Reduce physical exertion when AQI is in the moderate to poor range")

    if "Allergies" in health_conditions:
        recommendations.append("Take antihistamines preemptively during high pollen and pollution days")
        recommendations.append("Use HEPA air purifiers in your home")

    # Activity level recommendations
    if activity_level in ["Active", "Very Active"]:
        recommendations.append("Schedule outdoor workouts during times of better air quality")
        recommendations.append("Consider indoor exercise options during poor air quality days")
        recommendations.append("Monitor your breathing and heart rate during exercise")

    # Display recommendations
    for i, rec in enumerate(recommendations):
        st.markdown(f"**{i + 1}.** {rec}")

    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.header("🏙️ Urban Planning Insights")

    # Display zone analysis
    st.subheader("City Zone Analysis")

    # Create sample zone data
    zones = ['Industrial', 'Commercial', 'Residential', 'Green Spaces', 'Traffic Hubs']
    metrics = {
        'Average AQI': [120, 100, 80, 60, 110],
        'Peak AQI': [180, 150, 120, 90, 160],
        'Days Above Threshold': [15, 10, 5, 2, 12]
    }

    zone_data = pd.DataFrame(metrics, index=zones)
    st.table(zone_data)

    # Display recommendations
    st.subheader("Smart City Recommendations")

    st.markdown("""
    ### 🏭 Industrial Zone
    - Implement emission control systems
    - Create green buffer zones
    - Monitor emissions in real-time

    ### 🚦 Traffic Management
    - Smart traffic systems
    - Expand bicycle infrastructure
    - Improve public transport

    ### 🌳 Green Initiatives
    - Increase urban green spaces
    - Plant air-purifying trees
    - Create pedestrian zones
    """)

    # Display sustainability metrics
    st.subheader("Sustainability Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Green Space Ratio", "23%", "+3%")
        st.metric("Public Transport Usage", "45%", "+5%")

    with col2:
        st.metric("Air Quality Improvement", "15%", "+2%")
        st.metric("Clean Energy Adoption", "35%", "+8%")

    # Add real-time monitoring suggestion
    st.sidebar.markdown("""
    ---
    ### 📱 Stay Informed
    - Set up personalized AQI alerts
    - Monitor your exposure history
    - Track your health symptoms
    """)
