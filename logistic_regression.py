'''
This scrip implementats the logistic regression algorithem for classification.
'''
import numpy as np

class Logistic_regression :
    def __init__(self, learning_rate=0.1, n_epochs=1000):
        self.weights = None
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def sigmoid(self, x) :
        # Sigmoid activation function: maps input to a number in range(0, 1)
        return 1/(1 + np.exp(-x))

    def build(self, x, y) :
        x = np.c_[np.ones(x.shape[0]), x]  # Add bias column
        self.weights = np.zeros(x.shape[1]) # Initialize weights as zeros(with bias weight)
        # Gradient descent training loop
        for i in range(self.n_epochs) :
            prev_weights = self.weights.copy()
            # Stochastic Gradient Descent: update weights for each sample
            for xi, yi in zip(x,y) :
                # This calculates the gradiant of the log loss function at xi & yi
                grad = (self.sigmoid(self.weights.T @ xi) - yi ) * xi
                # Update weights using the gradient
                self.weights -= self.learning_rate * grad
            # Stop early if weights haven't changed significantly
            if np.allclose(self.weights, prev_weights) : break

    def predict(self, x, confidence=0.5):
        # Add bias term to input
        x = np.c_[np.ones(x.shape[0]), x]
        # Compute predicted probabilities
        p_predict = self.sigmoid(x @ self.weights)
        # Convert probabilities to class labels using a confidence threshold
        return np.where(p_predict >= confidence, 1, 0)
