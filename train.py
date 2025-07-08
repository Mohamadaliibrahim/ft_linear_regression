import pandas as pd
from linear_regression import LinearRegressionGD

df = pd.read_csv("data.csv")
model = LinearRegressionGD(lr=0.1, n_iter=10_000)
model.fit(df.km.values, df.price.values)
model.save()
print(f"Training done → θ0={model.beta0:.2f}, θ1={model.beta1:.6f}")
