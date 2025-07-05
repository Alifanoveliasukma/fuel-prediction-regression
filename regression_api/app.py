from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Fuel Prediction Regression API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    liter = data.get('liter')

    if liter is None:
        return jsonify({'error': 'Field \"liter\" is required'}), 400

    try:
        prediction = model.predict([[float(liter)]])
        return jsonify({
            'liter': liter,
            'prediksi_kilometer': round(float(prediction[0][0]), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

