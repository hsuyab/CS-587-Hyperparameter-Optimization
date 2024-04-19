import torch
import torch.nn as nn

class RidgeRegression(nn.Module):
    def __init__(self, theta, lambda_):
        super(RidgeRegression, self).__init__()
        self.theta = nn.Parameter(theta)
        self.lambda_ = lambda_
    
    def forward(self, X):
        return X.T @ self.theta
    
    def loss(self, y, y_pred, val=False):
        if val:
            return 0.5 * (torch.norm(y_pred - y) ** 2)
        else:
            return 0.5 * (torch.norm(y_pred - y) ** 2 + self.lambda_ * torch.norm(self.theta) ** 2)