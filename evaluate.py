import numpy as np, pandas as pd
from linear_regression import LinearRegressionGD

df = pd.read_csv("data.csv")
model = LinearRegressionGD.load()
pred = model.predict(df.km.values)
mse = np.mean((pred - df.price.values) ** 2)
r2  = 1 - mse / np.var(df.price.values)
print(f"MSE = {mse:.2f}   –   R² = {r2:.4f}")
