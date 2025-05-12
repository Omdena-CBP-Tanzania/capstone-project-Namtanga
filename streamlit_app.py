import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Set up the Streamlit app
st.set_page_config(page_title="Climate Prediction App", layout="wide")

# Load data and models
data_path = 'D/CAPSTONE_PROJECT/capstone-project-Namtanga/data/tanzania_climate_data_cleaned_dataset.csv'
model_path = 'D:/CAPSTONE_PROJECT/capstone-project-Namtanga/data/random_forest_model.pkl'
data = pd.read_csv(data_path)
model = joblib.load(model_path)

# Title and description
st.title("Climate Prediction App")
st.markdown("""
This interactive app allows you to explore climate trends and make predictions for future conditions in Tanzania.
You can visualize historical data, analyze trends, and input parameters for real-time predictions.
""")

# Sidebar for user input
st.sidebar.header("User Input Parameters")
year = st.sidebar.slider("Year", int(data['Year'].min()), int(data['Year'].max() + 10), int(data['Year'].min()))
month = st.sidebar.selectbox("Month", list(range(1, 13)))
rainfall = st.sidebar.number_input("Total Rainfall (mm)", float(data['Total_Rainfall_mm'].min()), float(data['Total_Rainfall_mm'].max()), float(data['Total_Rainfall_mm'].mean()))
max_temp = st.sidebar.number_input("Max Temperature (°C)", float(data['Max_Temperature_C'].min()), float(data['Max_Temperature_C'].max()), float(data['Max_Temperature_C'].mean()))
min_temp = st.sidebar.number_input("Min Temperature (°C)", float(data['Min_Temperature_C'].min()), float(data['Min_Temperature_C'].max()), float(data['Min_Temperature_C'].mean()))

# Prepare input for prediction
input_data = pd.DataFrame({
    'Year': [year],
    'Month': [month],
    'Total_Rainfall_mm': [rainfall],
    'Max_Temperature_C': [max_temp],
    'Min_Temperature_C': [min_temp]
})

# Prediction
st.sidebar.subheader("Prediction")
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.sidebar.write(f"Predicted Average Temperature (°C): {prediction:.2f}")

# Main Page - Visualizations
st.header("Exploratory Data Analysis")

# Time Series Plot
st.subheader("Temperature Trends Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['Year'] + data['Month'] / 12, data['Average_Temperature_C'], label="Avg Temp (°C)", color="blue")
ax.set_title("Average Temperature Over Time")
ax.set_xlabel("Year")
ax.set_ylabel("Temperature (°C)")
ax.legend()
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
correlation_matrix = data[['Average_Temperature_C', 'Total_Rainfall_mm', 'Max_Temperature_C', 'Min_Temperature_C']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Scatter Plot: Temperature vs Rainfall
st.subheader("Scatter Plot: Temperature vs Rainfall")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Average_Temperature_C', y='Total_Rainfall_mm', data=data, ax=ax)
ax.set_title("Temperature vs Rainfall")
ax.set_xlabel("Average Temperature (°C)")
ax.set_ylabel("Total Rainfall (mm)")
st.pyplot(fig)

# Seasonal Decomposition Visualization
st.subheader("Seasonal Patterns in Temperature")
data['YearMonth'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str))
data.set_index('YearMonth', inplace=True)
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['Average_Temperature_C'], model='additive', period=12)
result.plot()
st.pyplot(plt)
