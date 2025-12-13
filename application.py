import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. LOAD SAVED ASSETS
# ---------------------------------------------------------
# We use @st.cache_resource to load these only once, making the app faster
@st.cache_resource
def load_assets():
    model = joblib.load('best_pollution_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

# Load the raw data for the "Data Overview" page
# (Assuming you saved your cleaned data to csv earlier, if not, load the original raw file)
try:
    df = pd.read_csv('cleaned_air_quality_data.csv') # You might need to save this from your notebook
except:
    st.warning("cleaned_data.csv not found. Please save your dataframe to CSV in your notebook.")
    df = pd.DataFrame() # Empty placeholder

# ---------------------------------------------------------
# 2. SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.title("🇮🇳 India Air Quality Forecaster")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Data Explorer (EDA)", "Live Prediction"])

# ---------------------------------------------------------
# 3. PAGE: PROJECT OVERVIEW
# ---------------------------------------------------------
if page == "Project Overview":
    st.header("About the Project")
    st.write("""
    This application analyzes air quality data from 2015-2020 to predict **PM2.5** levels.
    
    **Key Features:**
    * **Data:** Historical records from 26 Indian cities.
    * **Model:** Random Forest Regressor (R2 Score: ~0.81).
    * **Goal:** Forecast pollution to aid public health decisions.
    """)
    
    if not df.empty:
        st.subheader("Dataset Sample")
        st.dataframe(df.head())
        st.write(f"**Total Records:** {df.shape[0]}")

# ---------------------------------------------------------
# 4. PAGE: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------
elif page == "Data Explorer (EDA)":
    st.header("Exploratory Data Analysis")
    
    if df.empty:
        st.error("Data not loaded.")
    else:
        # User Selection
        city = st.selectbox("Select a City to Analyze", df['City'].unique())
        
        # Filter Data
        city_data = df[df['City'] == city]
        
        # Plot 1: Trend Over Time
        st.subheader(f"Pollution Trends in {city}")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=city_data, x='Date', y='PM2.5', ax=ax, color='orange')
        ax.set_title(f"Daily PM2.5 Levels (2015-2020)")
        st.pyplot(fig)
        
        # Plot 2: Correlation Heatmap for that City
        st.subheader("Pollutant Correlations")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        numeric_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
        sns.heatmap(city_data[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

# ---------------------------------------------------------
# 5. PAGE: LIVE PREDICTION
# ---------------------------------------------------------
elif page == "Live Prediction":
    st.header("Predict PM2.5 Levels")
    st.write("Adjust the sliders below to simulate weather and pollution conditions.")
    
    # Input Widgets
    col1, col2 = st.columns(2)
    
    with col1:
        pm10 = st.slider("PM10 Level", 0, 500, 100)
        no2 = st.slider("NO2 Level", 0, 200, 40)
        so2 = st.slider("SO2 Level", 0, 100, 20)
        co = st.slider("CO Level", 0.0, 50.0, 1.0)
        
    with col2:
        o3 = st.slider("Ozone (O3)", 0, 200, 40)
        month = st.selectbox("Month", range(1, 13))
        year = st.number_input("Year", 2020, 2030, 2025)
        # Extract City names from the columns list (removing 'City_' prefix)
        city_options = [c.replace('City_', '') for c in model_columns if 'City_' in c]
        selected_city = st.selectbox("Select Location", city_options)

    # Prediction Logic
    if st.button("Predict Pollution Level"):
        # 1. Create a dictionary with all 0s for model columns
        input_data = {col: 0 for col in model_columns}
        
        # 2. Fill in the numeric values
        input_data['PM10'] = pm10
        input_data['NO'] = 25.0  # Assuming average if not input
        input_data['NO2'] = no2
        input_data['NOx'] = 30.0 # Assuming average
        input_data['NH3'] = 20.0 # Assuming average
        input_data['CO'] = co
        input_data['SO2'] = so2
        input_data['O3'] = o3
        input_data['Benzene'] = 3.0 # Assuming average
        input_data['Toluene'] = 8.0 # Assuming average
        input_data['Xylene'] = 2.0  # Assuming average
        input_data['Year'] = year
        input_data['Month'] = month
        
        # 3. Set the specific City column to 1 (One-Hot Encoding)
        city_col_name = f"City_{selected_city}"
        if city_col_name in input_data:
            input_data[city_col_name] = 1
            
        # 4. Convert to DataFrame and Scale
        input_df = pd.DataFrame([input_data])
        
        # Ensure columns are in the exact same order as training
        input_df = input_df[model_columns] 
        
        input_scaled = scaler.transform(input_df)
        
        # 5. Predict
        prediction = model.predict(input_scaled)[0]
        
        # 6. Display Result
        st.success(f"Predicted PM2.5 Concentration: {prediction:.2f} µg/m³")
        
        # Interpret the result
        if prediction < 60:
            st.info("Air Quality: Satisfactory 😊")
        elif prediction < 100:
            st.warning("Air Quality: Moderate 😐")
        else:
            st.error("Air Quality: Poor to Hazardous ☠️")