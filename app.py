from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Features order must match training data:
        # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        features = [
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age'])
        ]
        
        # Reshape and scale
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] # Probability of being positive
        
        # Prepare recommendation
        if prediction == 1:
            status = "High Risk"
            recommendation = "Our analysis suggests you may have a high risk of diabetes. Please consult a healthcare professional for a comprehensive evaluation."
        else:
            status = "Low Risk"
            recommendation = "Our analysis suggests you have a low risk of diabetes. Continue maintaining a healthy lifestyle with a balanced diet and regular exercise."

        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'risk_status': status,
            'probability': round(float(probability) * 100, 2),
            'recommendation': recommendation
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
