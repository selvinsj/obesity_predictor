import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import os

# Load dataset from the dataset folder
dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "generated_bmi_health_risks_dataset.csv")
df = pd.read_csv(dataset_path)

# Normalize column names to lowercase and strip whitespace
df.columns = df.columns.str.strip().str.lower()
print("Available columns:", df.columns.tolist())

# Use the exact column names from your dataset
weight_col = "weight (kg)"
height_col = "height (m)"
bmi_col = "bmi"
obesity_col = "obesity level"

# Encode categorical labels for Obesity Level
le = LabelEncoder()
df[obesity_col] = le.fit_transform(df[obesity_col])  # 0 for underweight, 1 for normal, 2 for overweight

# Select features and target
X = df[[weight_col, height_col, bmi_col]]
y = df[obesity_col]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the XGBoost model with tuned hyperparameters
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    use_label_encoder=False,
    eval_metric="mlogloss",
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save the model only if accuracy is 99% or above
model_save_path = os.path.join(os.path.dirname(__file__), "models", "bmi_model.pkl")
if accuracy >= 0.99:
    joblib.dump((model, scaler, le), model_save_path)
    print("✅ Model trained and saved successfully!")
else:
    print("⚠️ Accuracy below 99%. Try tuning hyperparameters!")
