import torch
import torch.nn as nn

class RidgeRegressionHG(nn.Module):
    def __init__(self, lambda_):
        super(RidgeRegressionHG, self).__init__()
        self.lambda_ = nn.Parameter(torch.tensor(lambda_, dtype=torch.float32))

    def forward(self, X, y):
        return torch.matmul(torch.linalg.inv(torch.matmul(X, X.T) + self.lambda_ * torch.eye(X.shape[0])), (torch.matmul(X, y)))

    def loss(self, y_val, X, theta):
        mse_loss = 0.5 * (torch.norm(y_val - X.T @ theta) ** 2)
        return mse_loss