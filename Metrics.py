import numpy as np


def mse(y_true, y_pred):
    """Tính Mean Squared Error (MSE)"""
    # Chuyển đổi sang mảng numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Tính trung bình bình phương sai số
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    """Tính Mean Absolute Error (MAE)"""
    # Chuyển đổi sang mảng numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Tính trung bình giá trị tuyệt đối của sai số
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """Tính R² (hệ số xác định)"""
    # Chuyển đổi sang mảng numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Tổng bình phương độ lệch so với giá trị trung bình
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Tổng bình phương sai số giữa giá trị thực tế và dự đoán
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    # Tính R² (1.0 = hoàn hảo, 0.0 = trung bình, <0.0 = tệ)
    r2 = 1 - (ss_residual / ss_total)
    
    return r2