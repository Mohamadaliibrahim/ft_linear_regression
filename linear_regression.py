import json
import numpy as np

class LinearRegressionGD:
    def __init__(self, lr=0.1, n_iter=10_000):
        self.lr, self.n_iter = lr, n_iter
        self.theta0 = self.theta1 = 0.0          # initial values
        self.mu = self.sigma = None              # for feature-scaling

    # ---------- helpers ----------
    def _normalize(self, x):
        self.mu, self.sigma = x.mean(), x.std()
        return (x - self.mu) / self.sigma

    # ---------- training ----------
    def fit(self, x_raw, y):
        x = self._normalize(x_raw.astype(float))
        m = len(x)
        for _ in range(self.n_iter):
            pred   = self.theta0 + self.theta1 * x
            error  = pred - y
            g0     = error.mean()
            g1     = (error * x).mean()
            self.theta0 -= self.lr * g0
            self.theta1 -= self.lr * g1

        # convert back to original units so we can predict directly with km
        self.beta1 = self.theta1 / self.sigma
        self.beta0 = self.theta0 - self.theta1 * self.mu / self.sigma

    # ---------- inference ----------
    def predict(self, x):
        return self.beta0 + self.beta1 * x

    # ---------- persistence ----------
    def save(self, path="model.json"):
        with open(path, "w") as f:
            json.dump({"theta0": self.beta0, "theta1": self.beta1}, f)

    @classmethod
    def load(cls, path="model.json"):
        with open(path) as f:
            p = json.load(f)
        m = cls(0, 0)
        m.beta0, m.beta1 = p["theta0"], p["theta1"]
        return m
