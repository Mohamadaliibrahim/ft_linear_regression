import numpy as np, pandas as pd
from linear_regression import LinearRegressionGD

df = pd.read_csv("data.csv")
model = LinearRegressionGD.load()
pred = model.predict(df.km.values)
errors = pred - df.price.values
m      = len(errors)
mse    = np.sum(errors ** 2) / m #mean square error
rmse = np.sqrt(mse) #root mse
var_y  = np.sum((df.price.values - df.price.values.mean()) ** 2) / m #variance of y
r2     = 1 - mse / var_y
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.4f}")