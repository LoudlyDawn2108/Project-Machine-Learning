import numpy as np
from RegressionTree import MyDecisionTreeRegressor

class MyRandomForestRegressor:
    """Random Forest: Ensemble nhiều Decision Trees, predict = average"""
    
    def __init__(self, n_estimators=100, max_depth=100, min_samples_split=2, min_samples_leaf=1, max_features=1.0, random_state=42):
        """
        Args:
            n_estimators: Số lượng trees trong forest
            max_depth: Độ sâu tối đa của mỗi tree
            min_samples_split: Số samples tối thiểu để split
            min_samples_leaf: Số samples tối thiểu ở leaf
            max_features: Số features tối đa để xem xét khi split (default 1.0 như sklearn)
                         None hoặc 1.0: dùng tất cả features
                         int: dùng max_features features
                         float (0.0 < x < 1.0): dùng max_features * n_features features
                         'sqrt': dùng sqrt(n_features) features
                         'log2': dùng log2(n_features) features
            random_state: Seed cho reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []  # List chứa các trees

    def fit(self, X, y):
        """Train forest: mỗi tree train trên bootstrap sample"""
        X = np.array(X) if not isinstance(X, np.ndarray) else X  # Chuyển về numpy array
        y = np.array(y).flatten() if not isinstance(y, np.ndarray) else y.flatten()  # Flatten y
        
        np.random.seed(self.random_state)  # Set seed để reproducible có thể tái tạo
        self.trees = []  # Reset list trees
        n_samples = X.shape[0]  # Số samples

        for _ in range(self.n_estimators):  # Train n_estimators trees
            # Bootstrap sampling: random chọn n_samples với replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]  # Lấy X theo indices
            y_sample = y[indices]  # Lấy y theo indices

            # Train 1 Decision Tree trên bootstrap sample
            tree = MyDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)  # Fit tree
            self.trees.append(tree)  # Thêm tree vào list

    def predict(self, X):
        """Predict = average predictions của tất cả trees"""
        predictions = np.array([tree.predict(X) for tree in self.trees])  # Lấy predictions từ tất cả trees
        
        return np.mean(predictions, axis=0)  # Trung bình theo axis=0 (theo từng sample)
    
    