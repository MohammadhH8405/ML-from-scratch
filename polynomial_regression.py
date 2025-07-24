"""
This script performs polynomial regression analysis.
the script generates data for usage altho that we could have used the data_loader function.
the script will plot the data if wanted.
"""
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def generate_data(n=50, noise=5.0):
    # Generates a polynomial with controllable noise 
    np.random.seed(42) # With this we will get the same data if we dont change anything with the function
    x = np.linspace(-10, 10, n)
    w = np.array([8, 5, -2, 2]) # Weights of the polynomial
    xb = np.vander(x, N=len(w) , increasing=True)
    random_noise_array = np.random.randn(n) * noise
    y = xb @ w + random_noise_array
    return x, y # Returns np.arrays

class Polynomial_regression() :
    def __init__(self):
        self.best_erms = float('inf')
        self.best_weights = None
        self.degree = None
        self.x = None

    def build(self, x, y, max_degree=10) :
        # Fits the best polynomial for the data, it uses test points erms to decide the degree of the polynomial.
        x = x.flatten() # Make sure the array given is 1D
        self.x = x
        # Split the data into 80% train and 20% test 
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # Go throgh different degrees (max 10) to find the best fit
        for i in range(1, max_degree + 1) :
            xb_train = np.vander(x_train, N=i + 1, increasing=True)
            w = np.linalg.pinv(xb_train.T @ xb_train) @ xb_train.T @ y_train
            xb_test = np.vander(x_test, N=i + 1, increasing=True)
            y_test_pred = xb_test @ w
            erms = np.sqrt(np.mean((y_test_pred - y_test) ** 2))
            # If we find a lower erms we know we have a better fit
            if erms < self.best_erms:
                self.best_erms = erms
                self.best_weights = w
        # Ignore the insignificant weights
        self.best_weights = np.array([0 if abs(coef) < 0.001 else coef for coef in self.best_weights])
        while len(self.best_weights) > 1 and self.best_weights[-1] == 0:
            self.best_weights = self.best_weights[:-1]
        self.degree = len(self.best_weights)-1

    def equation(self):
        # Make a string to better shows the polynomial
        poly_str = f'{self.best_weights[0]:.4f}'
        for power, coef in enumerate(self.best_weights[1:], start=1):
            if coef != 0: poly_str += f' + {coef:.4f}x^{power}'
        return poly_str

    def parameters(self) :
        return self.best_weights

    def predict(self, xin) :
        xin = np.vander(xin, N=self.degree + 1, increasing=True)
        return xin @ self.best_weights

    def plot(self):
        """
        Plot the polynomial fit and data.
        """
        x_sorted = np.sort(self.x)
        xb_sorted = np.vander(x_sorted, N=self.degree + 1, increasing=True)
        y_pred = xb_sorted @ self.best_weights
        # Plot data points
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', label='Data', alpha=0.5)
        plt.plot(x_sorted, y_pred, color='red', label=self.equation(), linewidth=2) # Plot polynomial fit
        # Labels and title
        plt.title('Polynomial Regression Fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    x, y = generate_data(n=1000, noise=5.0)
    model = Polynomial_regression()
    model.build(x, y)
    model.plot()
