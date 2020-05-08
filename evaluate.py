import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error

from math import sqrt

# return to optimize this one using Zach's solution

# def plot_residuals(y, yhat, yhat_baseline, df):
#     residual = yhat - y
#     residual_baseline = yhat_baseline - y
#     fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16, 9))
#     ax1 = axs[0]
#     ax2 = axs[1]
#     ax1.set_title("Residuals")
#     ax2.set_title("Baseline Residuals")
#     fig.text(.1, 0.5, "yhat - y", ha="center", va="center", rotation="vertical")
#     sns.scatterplot(x=y, y=residual, data=df, color="navy", ax=axs[0]) # residual
#     sns.scatterplot(x=y, y=residual_baseline, data=df, color="crimson", ax=axs[1]) # residual baseline
#     plt.show()

def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title(f'Actual vs Residual ({predicted.name})')
    return plt.gca()

def regression_errors(y, yhat, df):
    return pd.Series({
        "SSE": mean_squared_error(y, yhat)*df.shape[0],
        "ESS": sum((yhat - y.mean())**2),
        "TSS": sum((yhat - y.mean())**2) + mean_squared_error(y, yhat)*df.shape[0],
        "MSE": mean_squared_error(y, yhat),
        "RMSE": sqrt(mean_squared_error(y, yhat))
    })

# perhaps, add a baseline_calc argument that can be set to either mean or median so that logic can be added to the function to calculate the baseline within
def baseline_errors(y, yhat_baseline, df):
    return pd.Series({
        "SSE_baseline": mean_squared_error(y, yhat_baseline)*df.shape[0],
        "MSE_baseline": mean_squared_error(y, yhat_baseline),
        "RMSE_baseline": sqrt(mean_squared_error(y, yhat_baseline))
    })

def better_than_baseline(y, yhat, yhat_baseline, df):
    """
    Compares root mean square error of predictions and baseline to determine if the model is better than our baseline
    """
    RMSE = regression_errors(y, yhat, df)["RMSE"]
    RMSE_baseline = baseline_errors(y, yhat_baseline, df)["RMSE_baseline"]
    if RMSE < RMSE_baseline:
        return print("Model performs better than baseline")
    else:
        return print("Model does not predict better than baseline, and is therefore useless")

# return to optimize this one using Zach's solution
def model_significance(ols_model):
    """
    Takes in an ordinary least squares model and returns the r squared and p-value of the F-statistic
    """
    r2 = ols_model.rsquared
    f_pval = ols_model.f_pvalue
    return {
        "r^2 -- explained variance": r2,
        "p-value for model significance": f_pval
        }