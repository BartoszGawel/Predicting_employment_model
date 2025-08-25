from sklearn.linear_model import LinearRegression
import numpy as np

# employment forecast for the following month – the example refers to month no. 7, i.e. July
month = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  
employees = np.array([420, 210, 92, 210, 312, 301])  #You need to paste Your employment, this is an example for 6 month

#AI model
model = LinearRegression()
model.fit(month, employees)

# Forecast for July
forecast = model.predict([[7]])
rounded_forecast = round(float(forecast[0]))  
print(f"Forecast hire for next month: {rounded_forecast} employees")
