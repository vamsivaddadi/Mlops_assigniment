from flask import Flask, request, jsonify
import pandas as pd
import mlflow

app = Flask(__name__)

mlflow.set_tracking_uri("file:./mlruns")
MODEL_NAME = "Best_Iris_Model"
MODEL_VERSION = "1"  
print(f"Loading model '{MODEL_NAME}' version {MODEL_VERSION} from MLflow...")
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

TARGET_NAMES = ["Setosa", "Versicolor", "Virginica"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Invalid input, expected JSON with 'features' key"}), 400

        df = pd.DataFrame(data["features"], columns=[
            "sepal.length", "sepal.width", "petal.length", "petal.width"
        ])

        preds = model.predict(df)
        pred_labels = [TARGET_NAMES[int(p)] for p in preds]

        return jsonify({"predictions": pred_labels})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
