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
    w = np.array([8, 3, 2, 2]) # Weights of the polynomial
    xb = np.vander(x, N=len(w) , increasing=True)
    random_noise_array = np.random.randn(n) * noise
    y = xb @ w + random_noise_array
    return x, y # Returns np.arrays

def polynomial_regression(x, y, max_degree=10):
    # Fits the best polynomial for the data, it uses test points erms to decide the degree of the polynomial.
    x.flatten() # Make sure the array given is 1D
    # Split the data into 80% train and 20% test 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    best_erms = float('inf')
    best_weights = None
    # Go throgh different degrees (max 10) to find the best fir
    for i in range(1, max_degree + 1) :
        xb_train = np.vander(x_train, N=i + 1, increasing=True)
        w = np.linalg.pinv(xb_train.T @ xb_train) @ xb_train.T @ y_train

        xb_test = np.vander(x_test, N=i + 1, increasing=True)
        y_test_pred = xb_test @ w
        erms = np.sqrt(np.mean((y_test_pred - y_test) ** 2))
        # If we find a lower erms we know we have a better fit
        if erms < best_erms:
            best_erms = erms
            best_weights = w
    # Ignore the insignificant weights
    best_weights = [0 if abs(coef) < 0.001 else coef for coef in best_weights]
    while True :
        if best_weights[-1] == 0 :best_weights.pop()
        else: break
    # Make a string that better shows the polynomial
    poly_str = f'{best_weights[0]:.4f}'
    for power, coef in enumerate(best_weights[1:], start=1):
        if coef != 0: poly_str += f' + {coef:.4f}x^{power}'

    return {'degree': len(best_weights)-1, 'weights': best_weights,
            'Erms': best_erms, 'polynomial_string': f"y = {poly_str}"}

def plot_polynomial_fit(x, y):
    model = polynomial_regression(x, y)
    # Sort x for smooth plotting
    x_sorted = np.sort(x)
    
    # Rebuild the design matrix for sorted x
    degree = model['degree']
    xb_sorted = np.vander(x_sorted, N=degree + 1, increasing=True)
    
    # Predict y using the model weights
    y_pred = xb_sorted @ np.array(model['weights'])
    
    # Plot data points
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data', alpha=0.5)
    
    # Plot polynomial fit
    plt.plot(x_sorted, y_pred, color='red', label=model['polynomial_string'], linewidth=2)

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
    model = polynomial_regression(x, y, max_degree=10)
    print(model['polynomial_string'])
    plot_polynomial_fit(x, y)
