from sklearn.preprocessing import StandardScaler, MinMaxScaler
from functools import reduce
from flask import Flask, render_template, request
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
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

def unique(list1):
    ans = reduce(lambda re, x: re + [x] if x not in re else re, list1, [])
    print(ans)

app = Flask(__name__, static_url_path='')

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/train')
def train():
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

    data.to_csv("traffic_volume_data.csv", index=False)

    sns.set()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore')

    data = pd.read_csv("traffic_volume_data.csv")
    data = data.sample(n=min(10000, len(data)), replace=False).reset_index(drop=True)

    n1 = data['weather_type']
    n2 = data['weather_description']
    unique(n1)
    unique(n2)

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

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
    y_pred = regr.predict(testX)
    print('Mean Absolute Error:', mean_absolute_error(testY, y_pred))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ip = []
        is_holiday = request.form.get('is_holiday', 'no')
        ip.append(1 if is_holiday == 'yes' else 0)

        ip.append(int(request.form['temperature']))
        ip.append(int(request.form['day']))
        ip.append(int(request.form['time'][:2]))

        D = request.form['date']
        ip.append(int(D[8:]))  
        ip.append(int(D[:4]))  
        ip.append(int(D[5:7]))  

        s1 = request.form.get('x0', '')
        s2 = request.form.get('x1', '')

        ip.append((n1features.index(s1) + 1) if s1 in n1features else 0)
        ip.append((n2features.index(s2) + 1) if s2 in n2features else 0)

        scaled_input = x_scaler.transform([ip])
        prediction = regr.predict(scaled_input)
        y_pred = y_scaler.inverse_transform([prediction])[0][0]

        if y_pred <= 1000:
            traffic_status = "No Traffic"
        elif y_pred <= 3000:
            traffic_status = "Busy or Normal Traffic"
        elif y_pred <= 5500:
            traffic_status = "Heavy Traffic"
        else:
            traffic_status = "Worst Case"

        return render_template('output.html', data1=ip, op=y_pred, statement=traffic_status)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
