import pandas as pd
import numpy as np

# Simulating real-time sensor data
def generate_sensor_data(rows=1000):
    np.random.seed(42)
    data = {
        "sensor1": np.random.rand(rows) * 100,
        "sensor2": np.random.rand(rows) * 50,
        "sensor3": np.random.rand(rows) * 75,
        "sensor4": np.random.rand(rows) * 25,
        "failure": np.random.choice([0, 1], size=rows, p=[0.95, 0.05])  # 5% failure
    }
    return pd.DataFrame(data)

# Save data to CSV
df = generate_sensor_data()
df.to_csv("real_time_sensor_data.csv", index=False)
print("Sensor data generated successfully!")
