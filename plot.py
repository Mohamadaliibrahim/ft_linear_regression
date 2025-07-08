import numpy as np, pandas as pd, matplotlib.pyplot as plt
from linear_regression import LinearRegressionGD

df = pd.read_csv("data.csv")
model = LinearRegressionGD.load()

plt.scatter(df.km, df.price, label="data")
x_line = np.linspace(df.km.min(), df.km.max(), 100)
plt.plot(x_line, model.predict(x_line), label="regression")
plt.xlabel("Mileage (km)");  plt.ylabel("Price (€)")
plt.title("Mileage vs Price");  plt.legend();  plt.show()
