# app.py
from flask import Flask, render_template, request, jsonify
from models.predictive_model import PredictiveModel

app = Flask(__name__)

# Initialize the model
model = PredictiveModel('data/equipment_abnormality_detection_data_new.csv')
model.preprocess_data()
model.train_model()

@app.route('/')
def index():
    equipment_ids = model.data['equipment_id'].unique()
    return render_template('index.html', equipment_ids=equipment_ids)

@app.route('/results', methods=['POST'])
def results():
    equipment_id = int(float(request.form['equipment_id']))
    abnormalities = model.detect_abnormalities(equipment_id)
    forecast = model.get_forecast(equipment_id)
    return render_template('results.html', abnormalities=abnormalities.to_dict(orient='records'),
                           forecast=forecast, equipment_id=equipment_id, enumerate=enumerate)

app.run(debug=True)