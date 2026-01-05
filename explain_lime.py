import pandas as pd
import pickle
import lime.lime_tabular
import numpy as np

# Load dataset
df = pd.read_csv("real_time_sensor_data.csv")
X = df.drop(columns=["failure"])

# Load trained model
with open("models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X.values, feature_names=X.columns, class_names=["No Failure", "Failure"], mode="classification"
)

# Explain one instance
idx = np.random.randint(0, len(X))
exp = explainer.explain_instance(X.iloc[idx].values, model.predict_proba)
exp.show_in_notebook()
exp.save_to_file('lime_explanation.html')


