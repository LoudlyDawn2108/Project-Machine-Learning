import numpy as np

class TreeNode:
    """Node của cây: internal node (split) hoặc leaf node (predict)"""
    
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Args:
            feature_index: Feature nào dùng để split
            threshold: Ngưỡng split (x[feature] <= threshold → left, else → right)
            left: Cây con trái
            right: Cây con phải
            value: Giá trị dự đoán (chỉ có ở leaf node)
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class MyDecisionTreeRegressor:
    """Decision Tree: chia đệ quy để giảm phương sai"""
    
    def __init__(self, max_depth=100, min_samples_split=2, min_samples_leaf=1):
        """
        Args:
            max_depth: Giới hạn độ sâu cây
            min_samples_split: Số samples tối thiểu để chia
            min_samples_leaf: SSố samples tối thiểu ở lá
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        """Build cây bằng recursive splitting"""
        X = np.array(X) if not isinstance(X, np.ndarray) else X  # Chuyển về numpy array
        y = np.array(y).flatten() if not isinstance(y, np.ndarray) else y.flatten()  # Chuyển y từ dạng ma trận sang vector
        
        self.tree = self._build_tree(X, y)  # Xây dựng cây từ root

    def _build_tree(self, X, y, depth=0):
        """Xây dựng cây xử dụng đệ quy"""
        num_samples, num_features = X.shape  # Lấy số samples và features
        
        # Điều kiện dừng: không đủ samples hoặc đạt max_depth
        if num_samples < self.min_samples_split or depth >= self.max_depth:
            return self._calculate_leaf_value(y)  # Tạo leaf node

        best_split = self._get_best_split(X, y, num_features)  # Tìm split tốt nhất
        
        # Không tìm được split hợp lệ → tạo leaf
        if not best_split or best_split.get("variance_reduction", 0) <= 0:
            return self._calculate_leaf_value(y)

        # Chia data thành 2 nhánh
        left_indices = X[:, best_split["feature_index"]] <= best_split["threshold"]  # Samples đi left
        right_indices = X[:, best_split["feature_index"]] > best_split["threshold"]  # Samples đi right

        # Kiểm tra constraint min_samples_leaf
        if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
            return self._calculate_leaf_value(y)  # Không split nếu vi phạm constraint

        # Đệ quy build cây con
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)  # Build cây trái
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)  # Build cây phải

        # Trả về internal node
        return TreeNode(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_subtree,
            right=right_subtree
        )

    def _get_best_split(self, X, y, num_features):
        """Tìm split tốt nhất (variance reduction lớn nhất)"""
        best_split = {}  # Dict lưu thông tin split tốt nhất
        max_variance_reduction = -float("inf")  # Khởi tạo giá trị tối đa

        for feature_index in range(num_features):  # Thử từng feature
            feature_values = X[:, feature_index]  # Lấy values của feature
            unique_values = np.unique(feature_values)  # Lấy các giá trị unique
            
            if len(unique_values) == 1:  # Feature có 1 giá trị duy nhất
                continue  # Không thể split → bỏ qua
            
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2  # Midpoint giữa các giá trị liên tiếp
            
            for threshold in thresholds:  # Thử từng threshold
                left_indices = feature_values <= threshold  # Boolean mask cho left
                right_indices = feature_values > threshold  # Boolean mask cho right

                left_count = np.sum(left_indices)  # Đếm samples bên left
                right_count = np.sum(right_indices)  # Đếm samples bên right
                
                # Kiểm tra constraint min_samples_leaf
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue  # Bỏ qua split này

                # Tính variance reduction của split này
                variance_reduction = self._calculate_variance_reduction(
                    y, y[left_indices], y[right_indices]
                )

                # Cập nhật best split nếu tốt hơn
                if variance_reduction > max_variance_reduction:
                    max_variance_reduction = variance_reduction
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "variance_reduction": variance_reduction
                    }

        return best_split  # Trả về split tốt nhất

    def _calculate_variance_reduction(self, parent, left_child, right_child):
        """Tính độ giảm variance: Var(parent) - weighted Var(children)"""
        weight_left = len(left_child) / len(parent)  # Tỷ lệ mẫu bên trái
        weight_right = len(right_child) / len(parent)  # Tỷ lệ mẫu bên phải
        
        # Variance reduction = Var(parent) - weighted average Var(children)
        variance_reduction = self._variance(parent) - (
            weight_left * self._variance(left_child) + 
            weight_right * self._variance(right_child)
        )
        return variance_reduction
    
    def _variance(self, y):
        """Tính variance của y"""
        return np.var(y) if len(y) > 0 else 0  # Trả về 0 nếu y rỗng

    def _calculate_leaf_value(self, y):
        """Tạo nút lá    với giá trị = mean(y)"""
        return TreeNode(value=np.mean(y))  # Dự đoán = trung bình y
    
    def predict(self, X):
        """Dự đoán cho tất cả samples"""
        X = np.array(X) if not isinstance(X, np.ndarray) else X  # Chuyển về numpy array
        return np.array([self._predict_sample(sample, self.tree) for sample in X])  # Predict từng sample

    def _predict_sample(self, sample, tree):
        """Dự đoán 1 sample bằng cách đi xuống cây"""
        if tree.value is not None:  # Leaf node
            return tree.value  # Trả về giá trị dự đoán

        # Internal node: đi xuống left hoặc right
        if sample[tree.feature_index] <= tree.threshold:  # Điều kiện đi left
            return self._predict_sample(sample, tree.left)  # Đệ quy left
        else:
            return self._predict_sample(sample, tree.right)  # Đệ quy right