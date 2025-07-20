from linear_regression import LinearRegressionGD
model = LinearRegressionGD.load()
km = float(input("Mileage (km): "))
print(f"Estimated price ≈ {model.predict(km):.2f} $")
