"""
This module contains functions for visualizing the results of Ridge Regression and 
Ridge Regression with Hypergradients.
"""
from matplotlib import pyplot as plt
plt.style.use("fivethirtyeight")


# Plotting functions for Ridge Regression
def plot_lambda_vs_val_loss(lambda_vals, val_losses):
    """
    Plot lambda values against validation losses for Ridge Regression.

    Args:
        lambda_vals (list or numpy.ndarray): List or array of lambda values.
        val_losses (list or numpy.ndarray): List or array of validation losses.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(lambda_vals, val_losses)
    plt.xlabel("Lambda")
    plt.ylabel("Validation Loss")
    plt.title("Lambda vs Validation Loss")
    plt.savefig("img/lambda_vs_val_loss_ridge_regression.png")
    plt.show()


# Plotting functions for Ridge Regression with Hypergradients
def plot_loss_vs_iteration(val_losses):
    """
    Plot validation losses against iterations for Ridge Regression with Hypergradients.

    Args:
        val_losses (list or numpy.ndarray): List or array of validation losses.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(val_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs. Iteration (Ridge Regression)")
    plt.savefig("img/val_loss_vs_iterations_ridge_regression.png")
    plt.show()


# Plotting functions for Ridge Regression with lambda vs hypergradient iteration
def plot_lambda_vs_iteration(lambdas):
    """
    Plot lambda values against iterations for Ridge Regression with Hypergradients.

    Args:
        lambdas (list or numpy.ndarray): List or array of lambda values.
        model_type (str): Type of Ridge Regression model.
    """

    plt.figure(figsize=(10, 8))
    plt.plot(lambdas)
    plt.xlabel("Iteration")
    plt.ylabel("Lambda")
    plt.title("Lambda vs. Iteration (Ridge Regression)")
    plt.savefig("img/lambda_vs_iterations_ridge_regression.png")
    plt.show()
