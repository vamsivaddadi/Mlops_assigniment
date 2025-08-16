import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/processed/iris_processed.csv")
df['target'] = df['variety'].astype('category').cat.codes
target_names = df['variety'].astype('category').cat.categories

X = df.drop(columns=['variety', 'target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog()

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Iris_ML_Models")

best_model = None
best_acc = 0
best_model_name = ""

with mlflow.start_run(run_name="Logistic_Regression"):
    log_model = LogisticRegression(max_iter=200, random_state=42)
    log_model.fit(X_train, y_train)

    log_pred = log_model.predict(X_test)
    log_acc = accuracy_score(y_test, log_pred)

    mlflow.log_metric("accuracy", log_acc)
    mlflow.log_param("model_type", "LogisticRegression")

    if log_acc > best_acc:
        best_acc = log_acc
        best_model = log_model
        best_model_name = "LogisticRegression"


with mlflow.start_run(run_name="Random_Forest"):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    mlflow.log_metric("accuracy", rf_acc)
    mlflow.log_param("model_type", "RandomForest")

    if rf_acc > best_acc:
        best_acc = rf_acc
        best_model = rf_model
        best_model_name = "RandomForest"

import os
import joblib

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")


with mlflow.start_run(run_name="Best_Model_Registration") as run:
    print(f"Best Model: {best_model_name} with Accuracy: {best_acc:.4f}")
    mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="Best_Iris_Model")
