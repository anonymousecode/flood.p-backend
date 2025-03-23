from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the models and scaler
scaler = joblib.load("scaler.pkl")
neural_net = joblib.load("neural_network.pkl")
logistic_regression = joblib.load("logistic_regression.pkl")
gradient_boosting = joblib.load("gradient_boosting.pkl")

# Store models in a dictionary
models = {
    "Neural Network": neural_net,
    "Logistic Regression": logistic_regression,
    "Gradient Boosting": gradient_boosting
}

# Define feature names
train_features = [
    "MonsoonIntensity", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality",
    "IneffectiveDisasterPreparedness", "DrainageSystems",
    "CoastalVulnerability", "Landslides", "Watersheds", "WetlandLoss"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flood Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Convert JSON to DataFrame
        input_data = pd.DataFrame([data])
        
        # Ensure feature order
        input_data = input_data[train_features]
        
        # Scale input
        input_scaled = scaler.transform(input_data)

        # Generate predictions from all models
        predictions = {name: model.predict(input_scaled).tolist() for name, model in models.items()}
        
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
