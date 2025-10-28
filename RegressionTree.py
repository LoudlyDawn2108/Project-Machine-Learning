import numpy as np

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Chỉ số đặc trưng để chia
        self.threshold = threshold          # Ngưỡng để chia
        self.left = left                    # Cây con bên trái
        self.right = right                  # Cây con bên phải
        self.value = value                  # Giá trị dự đoán (nếu là lá)

class MyDecisionTreeRegressor:
    def __init__(self, max_depth=100, min_samples_split=2, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.random_state = random_state

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples < self.min_samples_split or depth >= self.max_depth:
            return self._calculate_leaf_value(y)

        best_split = self._get_best_split(X, y, num_features)
        if best_split["variance_reduction"] == 0:
            return self._calculate_leaf_value(y)

        left_indices = X[:, best_split["feature_index"]] <= best_split["threshold"]
        right_indices = X[:, best_split["feature_index"]] > best_split["threshold"]

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return TreeNode(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_subtree,
            right=right_subtree
        )

    def _get_best_split(self, X, y, num_features):
        best_split = {}
        max_variance_reduction = -float("inf")

        for feature_index in range(num_features):
            thresholds = set(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                variance_reduction = self._calculate_variance_reduction(y, y[left_indices], y[right_indices])

                if variance_reduction > max_variance_reduction:
                    max_variance_reduction = variance_reduction
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "variance_reduction": variance_reduction
                    }

        return best_split

    def _calculate_variance_reduction(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        variance_reduction = self._variance(parent) - (weight_left * self._variance(left_child) + weight_right * self._variance(right_child))
        return variance_reduction
    
    def _variance(self, y):
        return y.var() if len(y) > 0 else 0

    def _calculate_leaf_value(self, y):
        return TreeNode(value=y.mean())
    
    def predict(self, X):
        return [self._predict_sample(sample, self.tree) for sample in X]

    def _predict_sample(self, sample, tree):
        if tree.value is not None:
            return tree.value # Nút lá

        feature_index = tree.feature_index
        threshold = tree.threshold

        if sample[feature_index] <= threshold:
            return self._predict_sample(sample, tree.left)
        else:
            return self._predict_sample(sample, tree.right)