import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# --- Config ---
os.makedirs("models", exist_ok=True)
# Default to local file-based MLflow unless provided
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("Iris_Classification")

# --- Data ---
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Models ---
models = {
    "log_reg": LogisticRegression(max_iter=500, n_jobs=None),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
}

best_model = None
best_score = -1.0
best_run_id = None
best_name = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        # Fit
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log params/metrics
        mlflow.log_param("model_name", name)
        if name == "decision_tree":
            mlflow.log_param("max_depth", model.max_depth)
        if name == "log_reg":
            mlflow.log_param("max_iter", model.max_iter)

        mlflow.log_metric("accuracy", acc)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Track best
        if acc > best_score:
            best_score = acc
            best_model = model
            best_run_id = run.info.run_id
            best_name = name

# Save best locally
joblib.dump(best_model, "models/best_model.pkl")
print(f"[OK] Best model: {best_name} (accuracy={best_score:.4f}) -> models/best_model.pkl")

# --- (Optional) Register in MLflow Model Registry ---
# Works with file-based backend too; model versions stored in ./mlruns
model_uri = f"runs:/{best_run_id}/model"
model_name = os.getenv("MLFLOW_MODEL_NAME", "IrisClassifier")

try:
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"[OK] Registered model '{model_name}' v{registered.version} from run {best_run_id}")
    # Optionally transition stage
    from mlflow.tracking.client import MlflowClient
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"[OK] Transitioned '{model_name}' v{registered.version} to 'Staging'")
except Exception as e:
    print(f"[WARN] Model registry step skipped/failed: {e}")
