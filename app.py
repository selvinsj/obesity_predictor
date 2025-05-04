from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Load trained model components
model_path = os.path.join(os.path.dirname(__file__), "models", "bmi_model.pkl")
model, scaler, le = joblib.load(model_path)

# Load dataset for retrieving diet & exercise plans
dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "generated_bmi_health_risks_dataset.csv")
df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip().str.lower()  # normalize column names

# Define mapping from numeric label to text
obesity_mapping = {0: "Underweight", 1: "Normal", 2: "Overweight"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        weight = float(data["weight"])
        height = float(data["height"])
    except Exception as e:
        return jsonify({"error": "Invalid input"}), 400

    bmi = weight / (height ** 2)
    features = np.array([[weight, height, bmi]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    obesity_text = obesity_mapping.get(pred, "Unknown")

    # Retrieve a row from the dataset that matches the predicted obesity level (sample one row randomly)
    matching_rows = df[df["obesity level"] == pred]
    if matching_rows.empty:
        diet_plan = "No diet plan available."
        exercise_plan = "No exercise plan available."
        heart_risk = "N/A"
        hypertension_risk = "N/A"
    else:
        row = matching_rows.sample(n=1).iloc[0]
        diet_plan = row["diet plan"]
        exercise_plan = row["exercise plan"]
        heart_risk = row["heart attack risk (%)"]
        hypertension_risk = row["hypertension risk (%)"]

    return jsonify({
        "bmi": round(bmi, 2),
        "obesity_level": obesity_text,
        "diet_plan": diet_plan,
        "exercise_plan": exercise_plan,
        "heart_risk": heart_risk,
        "hypertension_risk": hypertension_risk
    })


if __name__ == "__main__":
    app.run(debug=True)
