import matplotlib.pyplot as plt
import numpy as np

class Knn :
    """
    K-Nearest Neighbors (KNN) algorithm for classification and regression.
    Parameters:
    x (ndarray) : Training data of shape (n_samples, n_features).
    y (ndarray) : Target values (labels or regression outputs).
    k (int, default=5) : Number of nearest neighbors to consider.
    distance_func (callable, optional) :
        A custom distance function. If None, Euclidean distance is used.
    """
    def __init__(self, x, y, k=5, distance_func=None):
        self.x = x
        self.y = y
        self.k = k
        self.accuracy = None
        self.sensitivity = None
        self.specificity = None
        self.precision = None
        if distance_func is None: self.distance_func = self._euclidean_distance
        else : self.distance_func = distance_func
    # Distance metrics
    def _euclidean_distance(self, x1, x2) :
        # Compute Euclidean distance between two vectors.
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2) :
        # Compute Manhattan (L1) distance between two vectors.
        return np.sum(np.abs(x1 - x2))

    def _minkowski_distance(self, x1, x2, p=3) :
        # Compute Minkowski distance of order p between two vectors.
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

    def _cosine_distance(self, x1, x2) :
        # Compute cosine distance between two vectors.
        cosine_simularity = np.sum(x1 * x2)/(np.sqrt(np.sum(x1**2))*np.sqrt(np.sum(x2**2)))
        return 1 - cosine_simularity

    def predict_class(self, new_x, is_weighted=False) :
        # This method will use the knn method for classification purposes.
        distances = [self.distance_func(i, new_x) for i in self.x]
        combined = np.column_stack((distances, self.y))
        sorted_combined = combined[combined[:, 0].argsort()]
        closest_points = sorted_combined[:self.k]
        if not is_weighted :
            labels = closest_points[:, 1].astype(int)
            return np.bincount(labels).argmax()
        else :
            label_weights = {}
            for i in closest_points:
                label = int(i[1])
                weight = 1 / (i[0] + 1e-10) # Small value to avoid zero division
                label_weights[label] = label_weights.get(label, 0) + weight
            return max(label_weights, key=label_weights.get)

    def predict_value(self, new_x, is_weighted=False) :
        # This method will use the knn method for regression purposes.
        distances = [self.distance_func(i, new_x) for i in self.x]
        combined = np.column_stack((distances, self.y))
        sorted_combined = combined[combined[:, 0].argsort()]
        closest_points = sorted_combined[:self.k]   
        if not is_weighted:
            return np.mean(closest_points[:, 1])
        else:
            w = 1 / (closest_points[:, 0] + 1e-10)
            v = closest_points[:, 1]
            return np.sum(w * v) / np.sum(w)

    def scores(self) :
        # Evaluate classification performance using accuracy, sensitivity (recall),specificity, and precision.
        TP = FP = TN = FN = 0
        for i, j in zip(self.x, self.y) :
            y_predict = self.predict_class(i)
            if j == y_predict and j == 1 : TP += 1
            if j != y_predict and j == 0 : FP += 1
            if j == y_predict and j == 0 : TN += 1
            if j != y_predict and j == 1 : FN += 1
        self.accuracy = (TP + TN)/ len(self.x)
        self.sensitivity = TP / (TP + FN)
        self.specificity = TN / (TN + FP)
        self.precision = TP / (TP + FP)

def plot_decision_boundaries(knn_model, title='KNN Decision Boundary'):
    """
    Plots the decision boundaries of a trained Knn classifier.
    Parameters:
    - knn_model: An instance of the Knn class that has already been trained.
    - title: Title of the plot.
    """
    X = knn_model.x
    y = knn_model.y
    
    # Define the range of the plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.1  # step size in the mesh

    # Create meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    # Predict for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = np.array([knn_model.predict_class(point) for point in grid_points])
    Z = predictions.reshape(xx.shape)
    # Plotting
    plt.figure(figsize=(10, 8))
    colors = plt.get_cmap('tab10', np.unique(y).size)
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=colors)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colors, edgecolor='k', s=40)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    unique_classes = np.unique(y) # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Class {cls}',
        markerfacecolor=colors(cls), markersize=10)for cls in unique_classes]
    plt.legend(handles=legend_handles, loc='upper right')
    plt.grid(True)
    plt.show()
