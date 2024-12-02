import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model

# Load models
@st.cache_resource
def load_ml_model():
    model = XGBRegressor()
    model.load_model('models/xg_model.pkl')
    return model

@st.cache_resource
def load_dl_model():
    return load_model('models/dl-model.keras')

ml_model = load_ml_model()
dl_model = load_dl_model()

# Page Title
st.title("House Price Prediction System")

# Input Form
with st.form("prediction_form"):
    st.write("### Enter House Details:")
    
    col1, col2 = st.columns(2)
    with col1:
        # Tract, Block Group, Sale Amount
        pin = st.number_input("PIN", value=123456)
        tract = st.number_input("Tract", value=1, min_value=1, step=1)
        bg = st.number_input("Block Group", value=1, min_value=1, step=1)
        sale_amt = st.number_input("Sale Amount ($)", value=100000, min_value=0, step=1000)

        # Other Inputs
        year_sold = st.number_input("Year Built *", value=2000, min_value=1800, max_value=2024, step=1)
        total_bath = st.number_input("Total Bathrooms *", value=1.0, min_value=0.5, max_value=10.0, step=0.5)
        bedrooms_nbr = st.number_input("Number of Bedrooms *", value=3, min_value=0, max_value=20, step=1)
        fireplaces = st.number_input("Fireplaces", value=0, min_value=0, max_value=10, step=1)
        garage = st.number_input("Garage Spaces", value=1, min_value=0, max_value=10, step=1)
        median_income = st.number_input("Median Income (K$) *", value=50, min_value=20, max_value=200, step=1)
        perc_18 = st.number_input("Percentage Under 18 (%)", value=20, min_value=0, max_value=100, step=1)

    with col2:
        living_space = st.number_input("Living Space (sq ft) *", value=1500, min_value=500, max_value=10000, step=10)
        total_rooms = st.number_input("Total Rooms *", value=6, min_value=1, max_value=20, step=1)
        house_age = st.number_input("House Age (years) *", value=10, min_value=0, max_value=100, step=1)
        lot_area = st.number_input("Lot Area (acres)", value=0.5, min_value=0.1, max_value=10.0, step=0.1)
        education_rate = st.number_input("High Education Rate (%)", value=50, min_value=0, max_value=100, step=1)
        distance_cbd = st.number_input("Distance to CBD (miles)", value=5, min_value=0, max_value=50, step=1)
        park_pct_5 = st.number_input("Park Percentage within 5 Miles (%)", value=30, min_value=0, max_value=100, step=1)
        park_pct_10 = st.number_input("Park Percentage within 10 Miles (%)", value=50, min_value=0, max_value=100, step=1)
        park_pct_15 = st.number_input("Park Percentage within 15 Miles (%)", value=70, min_value=0, max_value=100, step=1)

    submit_button = st.form_submit_button(label="Predict Price")

# Perform Prediction
if submit_button:
    # Prepare Input Data
    feature_data = pd.DataFrame({
        'pin':[pin],'tract': [tract], 'bg': [bg], 'SaleAmt': [sale_amt], 'YearSold': [year_sold],
        'TotalBath': [total_bath], 'BedroomsNbr': [bedrooms_nbr], 'FirePl': [fireplaces],
        'LivingSqFt': [living_space], 'RoomsNbr': [total_rooms], 'Age': [house_age],
        'LtArea': [lot_area * 43560],  # Convert acres to square feet
        'Garage': [garage], 'Med.Income': [median_income * 1000],  # Convert to dollars
        'Perc_18': [perc_18], 'Educ_High': [education_rate], 'Dist_cbd': [distance_cbd],
        'Park_Pct_5': [park_pct_5], 'Park_Pct_10': [park_pct_10], 'Park_Pct_15': [park_pct_15]
    })

    try:
        # Predictions
        ml_prediction = ml_model.predict(feature_data)[0]
        dl_prediction = dl_model.predict(feature_data.to_numpy())[0][0]

        # Display Results
        st.success(f"**XGBoost Model Prediction:** ${ml_prediction:,.2f}")
        st.success(f"**Deep Learning Model Prediction:** ${dl_prediction:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
