from flask import Flask, render_template, request, jsonify
import os
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load Dataset & Train Model at Startup
CSV_FILE = "respiratory_diseases_aqi.csv"

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError("Error: respiratory_diseases_aqi.csv not found. Ensure the file is in the working directory.")

df = pd.read_csv(CSV_FILE)

# Convert AQI range to numerical values
def aqi_to_numeric(aqi_range):
    match = re.search(r'(\d+)', str(aqi_range))
    return int(match.group(1)) if match else None

df['AQI_Lower_Bound'] = df['Critical AQI Levels'].apply(aqi_to_numeric)
df.dropna(subset=['AQI_Lower_Bound'], inplace=True)
df['AQI_Lower_Bound'] = df['AQI_Lower_Bound'].astype(int)

# Encode Disease Labels
label_encoder = LabelEncoder()
df['Disease_Encoded'] = label_encoder.fit_transform(df['Disease'])

# Train Random Forest Model
X = df[['AQI_Lower_Bound']]
y = df['Disease_Encoded']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction Function
def predict_disease(aqi_value):
    try:
        predicted_disease_encoded = model.predict([[aqi_value]])[0]
        predicted_disease = label_encoder.inverse_transform([predicted_disease_encoded])[0]

        disease_info = df[df['Disease'] == predicted_disease].iloc[0]

        return {
            'Disease': predicted_disease,
            'Effects': disease_info['Effects'],
            'Vulnerable_Groups': disease_info['Most Vulnerable Age Groups'],
            'Preventive_Measures': disease_info['Preventive Measures']
        }
    except Exception as e:
        return {'Error': f'Prediction failed: {str(e)}'}

# Webpage Route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    if request.method == 'POST':
        try:
            aqi_value = int(request.form['aqi'])
            prediction_result = predict_disease(aqi_value)
        except ValueError:
            prediction_result = {'Error': 'Invalid AQI value. Please enter a valid number.'}

    return render_template('index.html', prediction=prediction_result)

# API Route (for JSON-based responses)
@app.route('/api/predict', methods=['GET'])
def api_predict():
    try:
        aqi_value = int(request.args.get('aqi'))
        result = predict_disease(aqi_value)
        return jsonify(result)
    except ValueError:
        return jsonify({'Error': 'Invalid AQI value. Please enter a valid number.'})

# Run the Application
if __name__ == '__main__':
    app.run(debug=True)
