import pandas as pd
import joblib

clf, target_names, feature_names = joblib.load("decision_tree_model.pkl")


new_samples = pd.DataFrame([
    [5.1, 3.5, 1.4, 0.2],  
    [6.2, 3.4, 5.4, 2.3]   
], columns=feature_names)

predictions = clf.predict(new_samples)

predicted_labels = [target_names[p] for p in predictions]

for i, label in enumerate(predicted_labels):
    print(f"Sample {i+1} → Predicted class: {label}")
