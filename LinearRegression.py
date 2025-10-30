import numpy as np

class MyLinearRegression:
    """Linear Regression: y = w0 + w1*x1 + w2*x2 + ... + wn*xn"""
    
    def __init__(self, fit_intercept: bool = True) -> None:
        """
        Args: 
            fit_intercept: True = có hệ số chặn w0, False = không có w0
        """
        self.fit_intercept = fit_intercept  # Lưu cấu hình: có thêm hệ số chặn w0 hay không

    @property
    def coef_(self) -> np.ndarray:
        """Trả về hệ số [w1, w2, ..., wn] (không bao gồm w0)"""
        if not hasattr(self, 'w'):  # Kiểm tra đã train chưa
            raise ValueError("Model chưa được train")
        
        if self.fit_intercept:
            return self.w[1:]  # Bỏ w0, chỉ lấy [w1, w2, ...]
        return self.w  # Trả về toàn bộ nếu không có w0
    
    def fit(self, X, y) -> None:
        """
        Train model bằng Normal Equation: w = (X^T X)^(-1) X^T y
        
        Args:
            X: Ma trận features (n_samples, n_features)
            y: Vector target (n_samples,)
        """
        new_data = np.copy(X)  # Copy để không thay đổi dữ liệu gốc
        
        if self.fit_intercept:  # Nếu model có w0
            intercept = np.ones((new_data.shape[0], 1))  # Tạo cột 1 cho w0
            new_data = np.hstack((intercept, new_data))  # Ghép cột 1 vào đầu: [1, x1, x2, ...]

        self.data = new_data  # Lưu data đã thêm intercept
        
        # Normal Equation: w = (X^T X)^(-1) X^T y
        self.w = np.linalg.pinv(self.data.T @ self.data) @ self.data.T @ y  # Tính w = (X^T X)^(-1) X^T y
        # self.data.T: X^T
        # self.data: X
    def predict(self, x):
        """
        Dự đoán: y_pred = X @ w
        
        Args:
            x: Ma trận features (n_samples, n_features)
        Returns:
            np.ndarray: Vector dự đoán (n_samples,)
        """
        new_x = np.copy(x)  # Copy để không thay đổi dữ liệu gốc
        
        if self.fit_intercept:  # Nếu model có w0
            intercept = np.ones((new_x.shape[0], 1))  # Tạo cột 1 để khớp với w
            new_x = np.hstack((intercept, new_x))  # Ghép cột 1 vào đầu

        return new_x @ self.w  # Phép nhân ma trận: y = Xw
    
    