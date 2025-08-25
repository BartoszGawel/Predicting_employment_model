from sklearn.ensemble import RandomForestRegressor
import numpy as np

#Historical Data (employment)
month = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
employees = np.array([420, 210, 92, 210, 312, 301])

# Better Random Forest model for more complex data (big changes) Estimator = 100 means it create 100 trees.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(month, employees)

# Forecast for July
forecast = model.predict([[7]])
rounded_forecast = round(float(forecast[0]))

print(f"Forecast hire for next month using RandomForestRegressor: {rounded_forecast} employees")
