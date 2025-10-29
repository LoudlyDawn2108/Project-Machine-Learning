import numpy as np

class TreeNode:
    """Node của cây: internal node (split) hoặc leaf node (predict)"""
    
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Args:
            feature_index: Chỉ số của đặc trưng để chia
            threshold: Ngưỡng chia (x[feature] <= threshold → left, else → right)
            left: Cây con trái
            right: Cây con phải
            value: Giá trị dự đoán (chỉ có ở leaf node)
        """
        self.feature_index = feature_index  # Đặc trưng nào dùng để chia
        self.threshold = threshold  # Ngưỡng chia
        self.left = left  # Nút con trái
        self.right = right  # Nút con phải
        self.value = value  # Giá trị dự đoán (None nếu internal node)

class MyDecisionTreeRegressor:
    """Decision Tree: chia đệ quy để minimize variance"""
    
    def __init__(self, max_depth=100, min_samples_split=2, min_samples_leaf=1):
        """
        Args:
            max_depth: Độ sâu tối đa (tránh overfit)
            min_samples_split: Số samples tối thiểu để split node
            min_samples_leaf: Số samples tối thiểu ở leaf node
        """
        self.max_depth = max_depth  # Giới hạn độ sâu cây
        self.min_samples_split = min_samples_split  # Số mẫu tối thiểu để chia
        self.min_samples_leaf = min_samples_leaf  # Số mẫu tối thiểu ở lá
        self.tree = None  # Nút gốc của cây

    def fit(self, X, y):
        """Xây dựng cây bằng cách chia đệ quy"""
        X = np.array(X) if not isinstance(X, np.ndarray) else X  # Chuyển X về numpy array
        y = np.array(y).flatten() if not isinstance(y, np.ndarray) else y.flatten()  # Chuyển y từ dạng ma trận sang vector
        
        self.tree = self._build_tree(X, y)  # Xây dựng cây từ gốc

    def _build_tree(self, X, y, depth=0):
        """Xây dựng cây đệ quy"""
        num_samples, num_features = X.shape  # Lấy số mẫu và đặc trưng
        
        # Điều kiện dừng: không đủ mẫu hoặc đạt giới hạn độ sâu
        if num_samples < self.min_samples_split or depth >= self.max_depth:
            return self._calculate_leaf_value(y)  # Tạo leaf node

        best_split = self._get_best_split(X, y, num_features)  # Tìm ngưỡng chia tốt nhất

        # Không tìm được ngưỡng chia tốt nhất hợp lệ → tạo leaf
        if not best_split or best_split.get("variance_reduction", 0) <= 0:
            return self._calculate_leaf_value(y)

        # Chia dữ liệu thành tập dữ liệu của hai nút con
        left_indices = X[:, best_split["feature_index"]] <= best_split["threshold"]
        right_indices = X[:, best_split["feature_index"]] > best_split["threshold"]

        # Kiểm tra ràng buộc min_samples_leaf
        if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
            return self._calculate_leaf_value(y)  # Không chia nếu vi phạm ràng buộc

        # Đệ quy xây dựng cây con
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)  # Xây dựng cây trái
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)  # Xây dựng cây phải

        # Trả về internal node
        return TreeNode(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_subtree,
            right=right_subtree
        )

    def _get_best_split(self, X, y, num_features):
        """Tìm ngưỡng chia tốt nhất (độ giảm phương sai lớn nhất)"""
        best_split = {}  # Dict lưu thông tin ngưỡng chia tốt nhất
        max_variance_reduction = -float("inf")  # Khởi tạo độ giảm phương sai lớn nhất tìm được

        for feature_index in range(num_features):  # Thử từng đặc trưng
            feature_values = X[:, feature_index]  # Lấy giá trị dự đoán
            unique_values = np.unique(feature_values)  # Bỏ đi các giá trị trùng lặp

            if len(unique_values) == 1:  # Đặc trưng có 1 giá trị duy nhất
                continue  # Không thể split → bỏ qua
            
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2  # Midpoint giữa các giá trị liên tiếp
            
            for threshold in thresholds:  # Thử từng threshold
                left_indices = feature_values <= threshold  # Nút con bên trái có giá trị dự đoán nhỏ hơn hoặc bằng ngưỡng
                right_indices = feature_values > threshold  # Nút con bên phải có giá trị dự đoán lớn hơn ngưỡng

                left_count = np.sum(left_indices)  # Đếm mẫu bên trái
                right_count = np.sum(right_indices)  # Đếm mẫu bên phải

                # Kiểm tra ràng buộc min_samples_leaf
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue  # Bỏ qua ngưỡng chia này

                # Tính độ giảm phương sai của ngưỡng chia này
                variance_reduction = self._calculate_variance_reduction(
                    y, y[left_indices], y[right_indices]
                )

                # Cập nhật ngưỡng chia tốt nhất nếu tốt hơn
                if variance_reduction > max_variance_reduction:
                    max_variance_reduction = variance_reduction
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "variance_reduction": variance_reduction
                    }

        return best_split  # Trả về ngưỡng chia tốt nhất

    def _calculate_variance_reduction(self, parent, left_child, right_child):
        """Tính độ giảm variance: Var(parent) - weighted Var(children)"""
        weight_left = len(left_child) / len(parent)  # Tỷ lệ mẫu bên trái
        weight_right = len(right_child) / len(parent)  # Tỷ lệ mẫu bên phải
        
        # Độ giảm phương sai = Var(parent) - trung bình trọng số Var(children)
        variance_reduction = self._variance(parent) - (
            weight_left * self._variance(left_child) + 
            weight_right * self._variance(right_child)
        )
        return variance_reduction
    
    def _variance(self, y):
        """Tính phương sai của y"""
        return np.var(y) if len(y) > 0 else 0  # Trả về 0 nếu y rỗng

    def _calculate_leaf_value(self, y):
        """Tạo leaf node với giá trị = mean(y)"""
        return TreeNode(value=np.mean(y))  # Dự đoán = trung bình y
    
    def predict(self, X):
        """Dự đoán cho tất cả mẫu"""
        X = np.array(X) if not isinstance(X, np.ndarray) else X  # Chuyển về numpy array
        return np.array([self._predict_sample(sample, self.tree) for sample in X])  # Dự đoán từng mẫu

    def _predict_sample(self, sample, tree):
        """Dự đoán 1 mẫu bằng cách đi xuống cây"""
        if tree.value is not None:  # Leaf node
            return tree.value  # Trả về giá trị dự đoán

        # Internal node: đi xuống left hoặc right
        if sample[tree.feature_index] <= tree.threshold:  # Điều kiện đi left
            return self._predict_sample(sample, tree.left)  # Đệ quy left
        else:
            return self._predict_sample(sample, tree.right)  # Đệ quy right