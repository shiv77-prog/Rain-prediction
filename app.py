from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
loaded = joblib.load("rainfall_prediction_model.pkl")
model = loaded['model']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        cloud = float(data['cloud'])
        humidity = float(data['humidity'])
        pressure = float(data['pressure'])
        wind = float(data['wind_speed'])
        direction = float(data['direction'])
        dewpoint = float(data['dewpoint'])
        sunshine = float(data['sunshine'])

        # 7 features (IMPORTANT)
        features = np.array([[cloud, humidity, pressure, wind, direction, dewpoint, sunshine]])

        prediction = model.predict(features)

        result = "🌧 Rain Tomorrow" if prediction[0] == 1 else "☀ No Rain"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'prediction': f'Error: {str(e)}'})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)