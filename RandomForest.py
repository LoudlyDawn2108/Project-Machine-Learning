import numpy as np
from RegressionTree import MyDecisionTreeRegressor

class MyRandomForestRegressor:
    n_estimators: int
    max_depth: int
    random_state: int
    trees: list[MyDecisionTreeRegressor]

    def __init__(self, n_estimators=100, max_depth=100, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y).flatten() if not isinstance(y, np.ndarray) else y.flatten()
        np.random.seed(self.random_state)
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = MyDecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)