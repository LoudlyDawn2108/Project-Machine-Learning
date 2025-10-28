import numpy as np

class MyLinearRegression:
    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept

    @property
    def coef_(self) -> np.ndarray:
        if not hasattr(self, 'w'):
            raise ValueError("Model is not fitted yet")
        
        if self.fit_intercept:
            return self.w[1:]

        return self.w
    
    def fit(self, X, y) -> None:
        new_data = np.copy(X)
        if self.fit_intercept:
            intercept = np.ones((new_data.shape[0], 1))
            new_data = np.hstack((intercept, new_data))

        self.data = new_data
        self.w = np.linalg.pinv(self.data.T @ self.data) @ self.data.T @ y

    def predict(self, x):
        new_x = np.copy(x)
        if self.fit_intercept:
            intercept = np.ones((new_x.shape[0], 1))
            new_x = np.hstack((intercept, new_x))

        return new_x @ self.w