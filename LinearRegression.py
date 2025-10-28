import numpy as np

class MyLinearRegression:
    """
    Linear Regression tự implement sử dụng Normal Equation
    Công thức: y = w0 + w1*x1 + w2*x2 + ... + wn*xn
    """
    
    def __init__(self, fit_intercept: bool = True) -> None:
        """
        Khởi tạo model
        
        Args:
            fit_intercept: True nếu muốn có hệ số chặn w0
        """
        self.fit_intercept = fit_intercept

    @property
    def coef_(self) -> np.ndarray:
        """
        Trả về các hệ số w1, w2, ..., wn (không bao gồm w0)
        """
        if not hasattr(self, 'w'):
            raise ValueError("Model is not fitted yet")
        
        if self.fit_intercept:
            return self.w[1:]  # Bỏ w0, chỉ lấy w1, w2, ...

        return self.w
    
    def fit(self, X, y) -> None:
        """
        Train model bằng Normal Equation: w = (X^T * X)^(-1) * X^T * y
        
        Args:
            X: Ma trận đặc trưng (n_samples, n_features)
            y: Vector giá trị thực (n_samples,)
        """
        new_data = np.copy(X)
        
        if self.fit_intercept:
            # Thêm cột 1 vào đầu để tính w0
            intercept = np.ones((new_data.shape[0], 1))
            new_data = np.hstack((intercept, new_data))

        self.data = new_data
        # Áp dụng Normal Equation với pseudo-inverse
        self.w = np.linalg.pinv(self.data.T @ self.data) @ self.data.T @ y

    def predict(self, x):
        """
        Dự đoán giá trị y_pred = X * w
        
        Args:
            x: Ma trận đặc trưng cần dự đoán (n_samples, n_features)
            
        Returns:
            Vector giá trị dự đoán (n_samples,)
        """
        new_x = np.copy(x)
        
        if self.fit_intercept:
            # Thêm cột 1 để tương thích với w
            intercept = np.ones((new_x.shape[0], 1))
            new_x = np.hstack((intercept, new_x))

        return new_x @ self.w