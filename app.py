import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Global Variables
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
regr = MLPRegressor(random_state=1, max_iter=500)
n1features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist', 'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
n2features = ['light rain', 'few clouds', 'Sky is Clear', 'light snow', 'sky is clear', 'mist', 'broken clouds',
              'moderate rain', 'drizzle', 'overcast clouds', 'scattered clouds', 'haze', 'proximity thunderstorm',
              'light intensity drizzle', 'heavy snow', 'heavy intensity rain', 'fog', 'heavy intensity drizzle',
              'shower snow', 'snow', 'thunderstorm with rain', 'thunderstorm with heavy rain',
              'thunderstorm with light rain', 'proximity thunderstorm with rain', 'thunderstorm with drizzle',
              'smoke', 'thunderstorm', 'proximity shower rain', 'very heavy rain',
              'proximity thunderstorm with drizzle', 'light rain and snow', 'light intensity shower rain',
              'SQUALLS', 'shower drizzle', 'thunderstorm with light drizzle']

# --- Model Training Function ---
@st.cache_data
def train_model():
    data = pd.read_csv('static/Train.csv')
    data = data.sort_values(by=['date_time'], ascending=True).reset_index(drop=True)
    for n in [1, 2, 3, 4, 5, 6]:
        data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    data['is_holiday'] = data['is_holiday'].apply(lambda x: 0 if x == 'None' else 1)
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
    data['month_day'] = data['date_time'].dt.day
    data['weekday'] = data['date_time'].dt.weekday + 1
    data['month'] = data['date_time'].dt.month
    data['year'] = data['date_time'].dt.year

    n1 = data['weather_type']
    n2 = data['weather_description']

    n11 = [(n1features.index(val) + 1) if val in n1features else 0 for val in n1]
    n22 = [(n2features.index(val) + 1) if val in n2features else 0 for val in n2]

    data['weather_type'] = n11
    data['weather_description'] = n22

    features = ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month',
                'weather_type', 'weather_description']
    target = ['traffic_volume']

    X = x_scaler.fit_transform(data[features])
    y = y_scaler.fit_transform(data[target]).flatten()

    regr.fit(X, y)
    return regr, x_scaler, y_scaler

# Train the model (this will only run once and be cached)
model, x_scaler_trained, y_scaler_trained = train_model()

# --- Streamlit UI ---
st.title('Traffic Volume Prediction 🚗')

st.write("### Input the features to predict traffic volume:")

# --- Input Fields ---
is_holiday = st.selectbox('Is it a holiday?', ('no', 'yes'))
temperature = st.slider('Temperature (in Kelvin)', 250, 310, 280)
day = st.selectbox('Day of the week', (1, 2, 3, 4, 5, 6, 7))
time = st.slider('Time of the day (hour)', 0, 23, 12)
date = st.date_input("Date")

s1 = st.selectbox('Weather Type', n1features)
s2 = st.selectbox('Weather Description', n2features)


if st.button('Predict Traffic Volume'):
    # Prepare input for prediction
    ip = []
    ip.append(1 if is_holiday == 'yes' else 0)
    ip.append(temperature)
    ip.append(day)
    ip.append(time)
    
    # Extract date features
    ip.append(date.day)
    ip.append(date.year)
    ip.append(date.month)
    
    ip.append((n1features.index(s1) + 1) if s1 in n1features else 0)
    ip.append((n2features.index(s2) + 1) if s2 in n2features else 0)

    # Scale the input and predict
    scaled_input = x_scaler_trained.transform([ip])
    prediction = model.predict(scaled_input)
    y_pred = y_scaler_trained.inverse_transform([prediction])[0][0]

    # Display the result
    st.write(f"**Predicted Traffic Volume:** {int(y_pred)}")

    if y_pred <= 1000:
        traffic_status = "No Traffic"
    elif y_pred <= 3000:
        traffic_status = "Busy or Normal Traffic"
    elif y_pred <= 5500:
        traffic_status = "Heavy Traffic"
    else:
        traffic_status = "Worst Case"
        
    st.write(f"**Traffic Status:** {traffic_status}")
