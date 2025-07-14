"""
This script performs simple linear regression analysis
on 2D points data loaded from an Excel file.
It includes hypothesis testing and plotting of the data.
"""
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def data_loader(file) :
    """
    Reads the excel file and returns the names of the variables and their reletive data.
    """
    # Load data from Excel
    wb = openpyxl.load_workbook(file)
    ws = wb.active

    x_axis_name = ws['A1'].value
    y_axis_name = ws['B1'].value

    x_values = []
    y_values = []

    # Start from the second row (skip headers)
    for row in ws.iter_rows(min_row=2, max_col=2):
        x_cell, y_cell = row
        try:
            x = float(x_cell.value)
            y = float(y_cell.value)
            x_values.append(x)
            y_values.append(y)
        except (TypeError, ValueError):
            # Skip rows with invalid data
            continue
    wb.close()
    return x_axis_name, y_axis_name, x_values, y_values

def linear_regression(x_values, y_values) :
    """
    Performs simple linear regression and hypothesis testing.

    Parameters:
        x_values (list)
        y_values (list)

    Returns:
        A dictionary containing:
            - beta (float): Estimated slope.
            - alpha (float): Estimated intercept.
            - reject_beta (bool): Whether H0: beta = 0 is rejected.
            - reject_alpha (bool): Whether H0: alpha = 0 is rejected.
            - r2 (float): R-squared score shows how well the model fits the data.
            - mse (float): Mean squared error of the points.
    """
    n = len(x_values)
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    x_sum = sum(x_values)
    y_sum = sum(y_values)
    sum_xy = np.sum(x_values * y_values)
    sum_x2 = np.sum(x_values ** 2)

    # Calculate regression coefficients
    beta_hat = (sum_xy-(x_sum*y_sum/n)) / (sum_x2-(x_sum**2/n))
    alpha_hat = (y_sum / n) - (beta_hat*(x_sum / n))

    sxx = sum((i - (x_sum / n))**2 for i in x_values)
    syy = sum(((i - (y_sum / n))**2) for i in y_values)
    sxy = sum_xy - (x_sum*y_sum)/n

    mse = (syy - beta_hat*sxy)/(n-2)
    r2_score = 1 - ((syy - beta_hat*sxy)/syy)

    t_crit = stats.t.ppf(1-0.025, n-2)
    reject_beta_h0 = abs(beta_hat/np.sqrt(mse/sxx)) > t_crit
    reject_alpha_h0 = abs(alpha_hat/np.sqrt(mse*((1/n)+((x_sum / n)**2)/sxx))) > t_crit

    return {
        'beta': beta_hat,
        'alpha': alpha_hat,
        'reject_beta': reject_beta_h0,
        'reject_alpha': reject_alpha_h0,
        'r2': r2_score,
        'MSE' : mse }

def plot(x_axis_name, y_axis_name, x_values, y_values) :
    """
    Plots the data points and the fitted linear regression line.

    Parameters:
        x_axis_name (str): Label for the x-axis.
        y_axis_name (str): Label for the y-axis.
        x_values (list)
        y_values (list)
    """
    result = linear_regression(x_values, y_values)
    # Create regression line points
    x_min, x_max = min(x_values), max(x_values)
    x_range = np.linspace(x_min, x_max, 100)
    y_pred = [result['beta']*x + result['alpha'] for x in x_range]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the data points
    plt.scatter(x_values, y_values, color='blue', label='Data Points')

    # Plot the regression line
    text = f"Regression Line: y = {result['beta']:.3f}x + {result['alpha']:.3f}"
    plt.plot(x_range, y_pred, color='red', label=text)

    # Build annotation text
    info_text = (
        f"$R^2$ = {result['r2']:.3f}\n"
        f"MSE = {result['MSE']:.3f}\n"
        f"H₀: β = 0 → {'Rejected' if result['reject_beta'] else 'Not rejected'}\n"
        f"H₀: α = 0 → {'Rejected' if result['reject_alpha'] else 'Not rejected'}"
    )

    # Add text box to the plot
    plt.text(
        0.05, 0.95, info_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Add labels and title
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title('Linear Regression Analysis')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == '__main__' :
    xname, yname, x, y = data_loader('points.xlsx')
    plot(xname, yname, x, y)
