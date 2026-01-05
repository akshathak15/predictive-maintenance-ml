import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("real_time_sensor_data.csv")

# Prepare data
X = df.drop(columns=["failure"])
y = df["failure"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy:.2f}")

# Save model
with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("XGBoost Model Trained and Saved!")
