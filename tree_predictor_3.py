import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import graphviz

class TreeNode:
    """Represents a single node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None, samples=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
        self.samples = samples

    def is_leaf(self):
        """Check if the node is a leaf (no children)."""
        return self.label is not None

    def classify(self, sample):
        """Recursively classify a sample based on the split criteria."""
        if self.is_leaf():
            return self.label
        return self.left.classify(sample) if sample[self.feature] < self.threshold else self.right.classify(sample)

class DecisionTree(BaseEstimator):
    """Decision Tree Implementation from Scratch with Pruning."""
    def __init__(self, criterion="gini", max_depth=10, min_impurity_decrease=0.01):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None

    def fit(self, X, y):
        """Train the decision tree."""
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        if depth >= self.max_depth or len(set(y)) == 1:
            return TreeNode(label=Counter(y).most_common(1)[0][0], samples=y)
        
        best_feature, best_threshold, best_impurity = self._find_best_split(X, y)
        if best_feature is None or best_impurity < self.min_impurity_decrease:
            return TreeNode(label=Counter(y).most_common(1)[0][0], samples=y)
        
        left_mask = X[:, best_feature] < best_threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child, samples=y)

    def prune(self, X_val, y_val):
        """Post-pruning using Reduced Error Pruning on validation data."""

        def _prune_node(node):
            """Recursively prune the tree by checking validation accuracy."""
            if node.is_leaf():
                return

            _prune_node(node.left)
            _prune_node(node.right)

            if node.left.is_leaf() and node.right.is_leaf():
                y_pred_before = self.predict(X_val)
                acc_before = accuracy_score(y_val, y_pred_before)

                majority_class = Counter(node.samples).most_common(1)[0][0]
                original_feature, original_threshold = node.feature, node.threshold
                original_left, original_right = node.left, node.right
                node.feature = None
                node.threshold = None
                node.left = None
                node.right = None
                node.label = majority_class

                y_pred_after = self.predict(X_val)
                acc_after = accuracy_score(y_val, y_pred_after)

                if acc_after < acc_before:
                    node.feature = original_feature
                    node.threshold = original_threshold
                    node.left = original_left
                    node.right = original_right
                    node.label = None

        _prune_node(self.root)

    def predict(self, X):
        """Predict labels for given samples."""
        if self.root is None:
            raise ValueError("Decision tree has not been trained. Call `fit()` first.")
        return np.array([self.root.classify(sample) for sample in X])
    

    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        best_feature, best_threshold, best_impurity = None, None, float("inf")
        
        for feature in range(X.shape[1]):
            threshold = np.median(X[:, feature])
            left_mask = X[:, feature] < threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            impurity = self._compute_impurity(y[left_mask], y[right_mask])
            if impurity < best_impurity:
                best_feature, best_threshold, best_impurity = feature, threshold, impurity
        
        return best_feature, best_threshold, best_impurity

    def _compute_impurity(self, left_y, right_y):
        """Compute impurity using the chosen criterion, weighted by sample sizes."""
        n_left, n_right = len(left_y), len(right_y)
        n_total = n_left + n_right

        if n_total == 0:
            return 0

        if self.criterion == "gini":
            return (n_left / n_total) * self._gini_impurity(left_y) + (n_right / n_total) * self._gini_impurity(right_y)
        elif self.criterion == "entropy":
            return (n_left / n_total) * self._entropy(left_y) + (n_right / n_total) * self._entropy(right_y)
        elif self.criterion == "misclassification":
            return (n_left / n_total) * self._misclassification_error(left_y) + (n_right / n_total) * self._misclassification_error(right_y)

    def _gini_impurity(self, y):
        """Compute Gini impurity."""
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)

    def _entropy(self, y):
        """Compute entropy."""
        p = np.bincount(y) / len(y)
        return -np.sum(p * np.log2(p + 1e-9))

    def _misclassification_error(self, y):
        """Compute misclassification error."""
        p = np.bincount(y) / len(y)
        return 1 - np.max(p)

    def get_params(self, deep=True):
        """Get parameters."""
        return {"criterion": self.criterion, "max_depth": self.max_depth, "min_impurity_decrease": self.min_impurity_decrease}

    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

def zero_one_loss(y_true, y_pred):
    """Compute 0-1 loss."""
    return np.mean(y_true != y_pred)

def visualize_tree(tree, tree_name="Decision Tree"):
    """Generate a Graphviz visualization of the decision tree."""
    dot = graphviz.Digraph()
    
    def add_nodes_edges(dot, node, parent=None, edge_label=""):
        if node is None:
            return
        
        node_label = f"Feature: {node.feature}\nThreshold: {node.threshold:.2f}" if not node.is_leaf() else f"Leaf: {node.label}"
        dot.node(str(id(node)), node_label, shape="box" if not node.is_leaf() else "ellipse", style="filled", fillcolor="yellow" if not node.is_leaf() else "lightblue")
        
        if parent is not None:
            dot.edge(str(id(parent)), str(id(node)), label=edge_label)

        if not node.is_leaf():
            add_nodes_edges(dot, node.left, node, "True")
            add_nodes_edges(dot, node.right, node, "False")

    add_nodes_edges(dot, tree.root)
    return dot

def manual_train_test_split(X, y, test_size=0.2, seed=42):
    """Splits dataset into train and test sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def compute_accuracy(y_true, y_pred):
    """Computes accuracy."""
    return np.mean(y_true == y_pred)

def manual_cross_validation(model, X, y, k=5):
    """
    Performs k-fold cross-validation manually.
    Splits data into k subsets, trains the model on k-1 subsets,
    and tests on the remaining subset.
    """
    fold_size = len(X) // k
    scores = []
    
    for i in range(k):
        X_val = X[i * fold_size : (i + 1) * fold_size]
        y_val = y[i * fold_size : (i + 1) * fold_size]
        X_train = np.concatenate([X[:i * fold_size], X[(i + 1) * fold_size:]])
        y_train = np.concatenate([y[:i * fold_size], y[(i + 1) * fold_size:]])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(compute_accuracy(y_val, y_pred))

    return scores
