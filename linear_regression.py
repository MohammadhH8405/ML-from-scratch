"""
This script performs simple linear regression analysis
in this version the progam is able to work in multidimensional space.
It includes hypothesis testing and plotting of the data if possible.
"""
import openpyxl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats

def data_loader(file) :
    """
    Reads the excel file and returns the names of the variables and their reletive data.
    """
    # Load data from Excel
    wb = openpyxl.load_workbook(file)
    ws = wb.active

    headers = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
    x_values = []
    y_values = []

    for row in ws.iter_rows(min_row=2) : # Start from the second row (skip headers)
        x_cell, y_cell = row[:-1], row[-1]
        try:
            x = list(x_cell)
            x = [float(i.value) for i in x]
            y = float(y_cell.value)
            x_values.append(x)
            y_values.append(y)
        except (TypeError, ValueError):
            # Skip rows with invalid data
            continue
    wb.close()
    return headers, np.array(x_values), np.array(y_values)

def linear_regression(x_values, y_values) :
    """
    Performs simple linear regression and hypothesis testing.
    Parameters:
        x_values (numpy array)
        y_values (numpy array)
    Returns:
        A dictionary containing:
            - parameter vector (list with float)
            - reject_wj (list with bool) : Whether H0: wj = 0 is rejected.
            - r2 (float): R-squared score shows how well the model fits the data.
            - mse (float): Mean squared error of the points.
    """
    n =len(x_values)
    # Add a term for w0
    x = np.c_[np.ones((len(x_values), 1)), x_values]
    # Compute weights or the parameters vector (w = (X^T X)^-1 X^T y)
    w = np.linalg.inv(x.T @ x) @ x.T @ y_values
    y_predictor = x @ w
    # Calculate mse and r^2 score
    mse = (np.sum((y_predictor - y_values) ** 2))*(1/(n-2))
    ss_res = np.sum((y_values - y_predictor) ** 2)
    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
    r2_score = 1 - ss_res / ss_tot
    # Find the std of the parameter vector
    xtx_inv = np.linalg.inv(x.T @ x)
    std_w = np.sqrt(mse * np.diag(xtx_inv))
    # Find which parameters null hypothesis get rejected (H0 = 0)
    t_crit = stats.t.ppf(1 - 0.025, df=n-2)
    hypothesis_reject = [
    abs(w[i]/std_w[i]) > t_crit for i in range(len(w))]
    return {
        'parameter_vector' : w,
        'hyphthesis_reject' : hypothesis_reject,
        'r^2_score': r2_score,
        'MSE' : mse }

def plot(x_values, y_values, headers):
    """
    Plots the data points and the fitted regression model.
    Supports:
        - 2D plot when x_values has one feature
        - 3D surface plot when x_values has two features

    Parameters:
        x_values (np.array): Feature values (n_samples, n_features)
        y_values (np.array): Target values (n_samples,)
        headers (list): Column names from the Excel file
    """
    result = linear_regression(x_values, y_values)
    w = result['parameter_vector']
    r2 = result['r^2_score']
    mse = result['MSE']
    rejects = result['hyphthesis_reject']

    n_features = x_values.shape[1]

    # 2D LINE PLOT (1 feature)
    if n_features == 1:
        x_min, x_max = x_values.min(), x_values.max()
        x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        x_range_bias = np.c_[np.ones((100, 1)), x_range]
        y_pred = x_range_bias @ w

        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, color='blue', label='Data Points')
        plt.plot(x_range, y_pred, color='red', label=f"y = {w[1]:.3f}x + {w[0]:.3f}")

        info_text = (
            f"$R^2$ = {r2:.3f}\n"
            f"MSE = {mse:.3f}\n"
            f"H₀: w₀ = 0 → {'Rejected' if rejects[0] else 'Not rejected'}\n"
            f"H₀: w₁ = 0 → {'Rejected' if rejects[1] else 'Not rejected'}")

        plt.text(0.05, 0.95, info_text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.xlabel(headers[0])
        plt.ylabel(headers[-1])
        plt.title("2D Linear Regression")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 3D SURFACE PLOT (2 features)
    elif n_features == 2:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        x1 = x_values[:, 0]
        x2 = x_values[:, 1]
        ax.scatter(x1, x2, y_values, color='blue', label='Data Points')

        # Create grid for surface
        x1_range = np.linspace(x1.min(), x1.max(), 30)
        x2_range = np.linspace(x2.min(), x2.max(), 30)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        x1x2_flat = np.c_[np.ones(x1_grid.size), x1_grid.ravel(), x2_grid.ravel()]
        y_pred_flat = x1x2_flat @ w
        y_grid = y_pred_flat.reshape(x1_grid.shape)

        ax.plot_surface(x1_grid, x2_grid, y_grid, color='red', alpha=0.5, label="Regression Surface")

        ax.set_xlabel(headers[0])
        ax.set_ylabel(headers[1])
        ax.set_zlabel(headers[-1])
        ax.set_title("3D Linear Regression")

        plt.tight_layout()
        plt.show()

    # ❌ MORE THAN 2 FEATURES
    else:
        print("❌ Cannot plot when there are more than 2 input features.")
        print("Use dimensionality reduction (e.g., PCA) or drop features for visualization.")

if __name__ == '__main__':
    headers, x, y = data_loader('points.xlsx')
    linear_regression(x, y)
    plot(x, y, headers)
