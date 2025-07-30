import matplotlib.pyplot as plt
import numpy as np

class Perceptron :
    """
    A simple implementation of the Perceptron algorithm for binary classification.
    Parameters:
    learning_rate:(float) The step size used to update weights (default: 0.1)
    n_epochs:(int) The maximum number of passes through the training data (default: 1000)
    """
    def __init__(self, learning_rate=0.1, n_epochs=1000):
        self.bias = 0.0
        self.weights = None
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.accuracy = None
        self.sensitivity = None
        self.specificity = None
        self.percision = None

    def build(self, x, y) :
        """
        Train the Perceptron on the input data.
        Parameters:
        x : ndarray of shape (n_samples, n_features) Input features
        y : ndarray of shape (n_samples,) Target labels (0 or 1)
        """
        self.weights = np.zeros(x.shape[1])
        for i in range(self.n_epochs) :
            errors = 0
            for ind, val in enumerate(x) :
                # Predict using the current weights and bias
                y_predicted = np.where((val @ self.weights + self.bias) >= 0 , 1, 0)
                # Update the weights and the bias if the predection was wrong
                if y_predicted != y[ind] :
                    errors += 1
                    update = self.learning_rate * (y[ind] - y_predicted)
                    self.weights += update * val
                    self.bias += update
            if errors == 0 : # The model is good to use if there are no errors
                break
        # Some metrics to analys the performans of the model
        TP = FP = TN = FN = 0
        for i, j in zip(x, y) :
            y_predict = self.predict(i.reshape(1, -1)).ravel()[0]
            if j == y_predict and j == 1 : TP += 1
            if j != y_predict and j == 0 : FP += 1
            if j == y_predict and j == 0 : TN += 1
            if j != y_predict and j == 1 : FN += 1
        self.accuracy = (TP + TN)/ len(x)
        self.sensitivity = TP / (TP + FN)
        self.specificity = TN / (TN + FP)
        self.percision = TP / (TP + FP)

    def predict(self, x) :
        """
        Predict class labels for samples in x.
        Parameters:
        x : ndarray of shape (n_samples, n_features) Input samples
        Returns:
        ndarray of shape (n_samples,) Predicted class labels (0 or 1)
        """
        g_predict = x @ self.weights + self.bias
        return np.where(g_predict >= 0 , 1, 0)

def plot_decision_boundary(x, y):
    """
    Plot the data points and decision boundary.
    Parameters:
    x : ndarray of shape (n_samples, 2) 2D input features
    y : ndarray of shape (n_samples,) Target labels
    """
    if x.shape[1] != 2:
        raise ValueError("Plotting is only supported for 2D features.")
    model = Perceptron()
    model.build(x, y)
    # Scatter plot of data points
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='blue', label='Class 1')
    # Decision boundary: w1*x1 + w2*x2 + b = 0 => x2 = (-w1*x1 - b) / w2
    x_vals = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    y_vals = (-model.weights[0] * x_vals - model.bias) / model.weights[1]
    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    plt.xlim(np.min(x[:, 0]) - 1, np.max(x[:, 0]) + 1)
    plt.ylim(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()
