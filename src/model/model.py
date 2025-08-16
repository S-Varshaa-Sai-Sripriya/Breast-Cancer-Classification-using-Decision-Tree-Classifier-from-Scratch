import numpy as np
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys


class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        parent_entropy = self.entropy(y)
        best_gain = 0
        best_feature, best_threshold = None, None

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = np.where(X[:, feature] <= threshold)
                right_idx = np.where(X[:, feature] > threshold)

                if len(left_idx[0]) == 0 or len(right_idx[0]) == 0:
                    continue

                left_entropy = self.entropy(y[left_idx])
                right_entropy = self.entropy(y[right_idx])
                child_entropy = (len(left_idx[0]) * left_entropy + len(right_idx[0]) * right_entropy) / m
                info_gain = parent_entropy - child_entropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)

        feature, threshold = self.best_split(X, y)
        if feature is None:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)

        left_idx = np.where(X[:, feature] <= threshold)
        right_idx = np.where(X[:, feature] > threshold)
        left_child = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        return DecisionTreeNode(feature, threshold, left_child, right_child)

    def fit(self, X, y):
        try:
            logger.info("Training Decision Tree Classifier from scratch...")
            self.root = self.build_tree(X, y)
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error("Error during training")
            raise CustomException(e, sys) from e

    def predict_one(self, inputs, node):
        if node.value is not None:
            return node.value
        if inputs[node.feature] <= node.threshold:
            return self.predict_one(inputs, node.left)
        else:
            return self.predict_one(inputs, node.right)

    def predict(self, X):
        return np.array([self.predict_one(inputs, self.root) for inputs in X])
