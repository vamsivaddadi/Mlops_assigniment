from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os



import logging

logging.basicConfig(filename="logs/predictions.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s")


model_path = "models/best_model.pkl"
print(f"Loading model from {model_path}...")
model = joblib.load(model_path)
print("Model loaded successfully!")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ML Model Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)

        return jsonify({
            "input": data,
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
