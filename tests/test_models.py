"""
Module for testing models and datasets.
"""
import unittest
import torch
from models.ridge_regression import RidgeRegression
from models.ridge_regression_hg import RidgeRegressionHG
from datasets.diabetes_dataset import load_diabetes_data



class TestRidgeRegression(unittest.TestCase):
    """
    Test class for RidgeRegression.
    """
    def test_forward_pass(self):
        theta = torch.randn(2, 1)  # Dx1
        X = torch.randn(2, 3)  # DxN
        model = RidgeRegression(theta, lambda_=0.1)
        y_pred = model(X).detach()
        self.assertEqual(y_pred.shape, (3, 1))

    def test_loss_calculation_shape(self):
        """
        Test loss calculation shape of RidgeRegression.
        """
        theta = torch.randn(2, 1)
        X = torch.randn(2, 3)
        y = torch.randn(3, 1)
        model = RidgeRegression(theta, lambda_=0.1)
        y_pred = model(X)
        loss = model.loss(y, y_pred)
        self.assertEqual(loss.shape, torch.Size([]))


class TestRidgeRegressionHG(unittest.TestCase):
    """
    Test class for RidgeRegressionHG. (Hypergradient))
    """
    def setUp(self):
        torch.manual_seed(0)
        self.X = torch.randn(2, 3)
        self.y = torch.randn(3, 1)
        self.theta = torch.randn(2, 1)
        self.lambda_ = 0.0
        self.model = RidgeRegressionHG(self.lambda_)

    def test_forward_pass(self):
        """
        Test forward pass of RidgeRegressionHG.
        """
        theta = self.model(self.X, self.y)
        expected_theta = torch.linalg.inv(self.X @ self.X.T) @ (self.X @ self.y)
        self.assertTrue(torch.allclose(theta, expected_theta))

    def test_loss_calculation(self):
        """
        Test loss calculation of RidgeRegressionHG.
        """
        y = torch.tensor([[1], [2]], dtype=torch.float32)
        theta = torch.tensor([[1], [1]], dtype=torch.float32)
        X = torch.tensor([[0.5, 1], [0.5, 1]], dtype=torch.float32)
        lambda_ = 0.0
        model = RidgeRegressionHG(lambda_)
        loss = model.loss(y, X, theta).item()
        self.assertEqual(loss, 0.0)


class TestLoadDiabetesData(unittest.TestCase):
    """
    Test class for load_diabetes_data.
    """
    def test_load_diabetes_data(self):
        """
        Test load_diabetes_data function.
        """
        X_train, y_train, X_val, y_val = load_diabetes_data()

        # Check data types
        self.assertIsInstance(X_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertIsInstance(X_val, torch.Tensor)
        self.assertIsInstance(y_val, torch.Tensor)

        # Check data shapes
        self.assertEqual(X_train.shape[0], 10)  # Number of features
        self.assertEqual(X_train.shape[1], 353)  # Number of training samples
        self.assertEqual(y_train.shape[0], 353)  # Number of training targets
        self.assertEqual(y_train.shape[1], 1)  # Target dimension
        self.assertEqual(X_val.shape[0], 10)  # Number of features
        self.assertEqual(X_val.shape[1], 89)  # Number of validation samples
        self.assertEqual(y_val.shape[0], 89)  # Number of validation targets
        self.assertEqual(y_val.shape[1], 1)  # Target dimension

        # Check data types
        self.assertEqual(X_train.dtype, torch.float32)
        self.assertEqual(y_train.dtype, torch.float32)
        self.assertEqual(X_val.dtype, torch.float32)
        self.assertEqual(y_val.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
