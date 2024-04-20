"""
This script implements Ridge Regression with Hypergradient Descent.
"""
import torch
from torch import nn


class RidgeRegressionHG(nn.Module):
    """
    Ridge Regression with Hypergradient Descent.

    This class implements Ridge Regression with Hypergradient Descent, 
    which is a linear regression model regularized by the L2 norm (ridge penalty). 
    It uses hypergradient descent to optimize the regularization
    parameter lambda.

    Args:
        lambda_ (float): The regularization parameter.

    Attributes:
        lambda_ (torch.Tensor): The regularization parameter as a trainable parameter.

    Methods:
        forward(X, y): Performs forward pass of the ridge regression model.
        loss(y_val, X, theta): Calculates the mean squared error loss.

    """

    def __init__(self, lambda_):
        super(RidgeRegressionHG, self).__init__()
        self.lambda_ = nn.Parameter(torch.tensor(lambda_, dtype=torch.float32))

    def forward(self, X, y):
        """
        Performs the forward pass of the ridge regression model.
        Args:
            X (torch.Tensor): The input features.
            y (torch.Tensor): The target values.
        Returns:
            theta_star (torch.Tensor): The optimized model parameters.
        """        
        return torch.matmul(
            torch.linalg.inv(
                torch.matmul(X, X.T) + self.lambda_ * torch.eye(X.shape[0])
            ),
            (torch.matmul(X, y)),
        )

    def loss(self, y_val, X, theta):
        """
        Calculates the mean squared error (MSE) loss for ridge regression.

        Parameters:
        - y_val (torch.Tensor): The target values.
        - X (torch.Tensor): The input features.
        - theta (torch.Tensor): The model parameters.

        Returns:
        - mse_loss (torch.Tensor): The calculated MSE loss.
        """
        mse_loss = 0.5 * (torch.norm(y_val - X.T @ theta) ** 2)
        return mse_loss
