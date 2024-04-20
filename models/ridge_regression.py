import torch
from torch import nn


class RidgeRegression(nn.Module):
    """
    Ridge Regression model for linear regression with L2 regularization.
    """

    def __init__(self, input_dim):
        """
        Initialize the RidgeRegression model.

        Args:
            theta (torch.Tensor): The weight parameters of the model.
            lambda_ (float): The regularization parameter.
        """
        super().__init__()
        # self.theta = nn.Parameter(theta)
        # self.lambda_ = lambda_
        self.linear = nn.Linear(input_dim, 1, bias=False)
    def forward(self, X):
        """
        Perform forward pass of the RidgeRegression model.

        Args:
            X (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The predicted output.
        """
        return self.linear(X)

    # def loss(self, y, y_pred, val=False):
    #     """
    #     Calculate the loss function of the RidgeRegression model.

    #     Args:
    #         y (torch.Tensor): The ground truth labels.
    #         y_pred (torch.Tensor): The predicted labels.
    #         val (bool, optional): Whether the loss is for validation. Defaults to False.

    #     Returns:
    #         torch.Tensor: The calculated loss.
    #     """
    #     if val:
    #         return 0.5 * (torch.norm(y_pred - y) ** 2)
        
    #     return 0.5 * (
    #             torch.norm(y_pred - y) ** 2 + self.lambda_ * torch.norm(self.theta) ** 2
    #         )
