from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("iris_model.pkl")

# Home route - shows the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route for form submission from HTML
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # Get values from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=f"Predicted Iris species: {prediction}")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

# API route for JSON input
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

# Run the app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
