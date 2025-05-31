from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Iris Model API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
