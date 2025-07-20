import csv
import numpy as np

class LinearRegressionGD:
    def __init__(self, lr=0.1, n_iter=10000):
        self.lr, self.n_iter = lr, n_iter       # lr : learning rate, n_inter : max nb of lr
        self.theta0 = self.theta1 = 0.0
        self.mu = self.sigma = None

    def to_sd(self, x):
        self.mu = x.mean()                      #mean of x
        self.sigma =  x.std()                   #standard deviation
        return (x - self.mu) / self.sigma

    def fit(self, x_raw, y):
        x = self.to_sd(x_raw.astype(float))     #convert data unit to standard deviation unit
        for i in range(self.n_iter):
            pred   = self.theta0 + self.theta1 * x
            error  = pred - y
            fix0     = error.mean()
            fix1     = (error * x).mean()       #Gradient Descent
            self.theta0 -= self.lr * fix0
            self.theta1 -= self.lr * fix1

        # convert back to original units so we can predict directly with km
        self.beta1 = self.theta1 / self.sigma
        self.beta0 = self.theta0 - self.theta1 * self.mu / self.sigma

    def predict(self, x):
        return self.beta0 + self.beta1 * x

    def save(self, path="model.csv"):
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["theta0", "theta1"])
            writer.writerow([self.beta0, self.beta1])

    @classmethod
    def load(cls, path="model.csv"):
        with open(path, mode="r") as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
            beta0, beta1 = map(float, row)

        m = cls(0, 0)
        m.beta0 = beta0
        m.beta1 = beta1
        return m
