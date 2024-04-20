"""Implements the main function for the ridge regression model."""

import argparse
import torch
from torch import optim
from tqdm import tqdm

# from datasets import load_diabetes_data
from datasets.diabetes_dataset import load_diabetes_data
from models.ridge_regression import RidgeRegression
from models.ridge_regression_hg import RidgeRegressionHG
from utils.visualize import (
    plot_loss_vs_iteration,
    plot_lambda_vs_iteration,
    plot_lambda_vs_val_loss,
)


# Add command-line arguments
parser = argparse.ArgumentParser(description="Ridge Regression")
parser.add_argument(
    "--model",
    type=str,
    default="normal",
    choices=["normal", "hg"],
    help="Model type (default: normal)",
)
parser.add_argument(
    "--lambda_init",
    type=float,
    default=0.5,
    help="Initial value of lambda (default: 0.5)",
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
)
parser.add_argument(
    "--num_epochs", type=int, default=15001, help="Number of epochs (default: 15001)"
)
args = parser.parse_args()


def main():
    """
    Main function for ridge regression model.
    
    This function loads the diabetes dataset, initializes the ridge regression model,
    and trains the model using either normal ridge regression or ridge regression with
    hypergradient descent. It then plots the validation loss against the lambda values
    for normal ridge regression and the validation loss against the iterations for
    ridge regression with hypergradient descent.
    """
    X_train, y_train, X_val, y_val = load_diabetes_data()

    # selecting the normal model
    if args.model == "normal":
        theta = torch.randn(X_train.shape[0], 1, requires_grad=True)
        lambda_vals = [0, 0.1, 1, 10, 100]
        # model = RidgeRegression(lambda_=args.lambda_init)
        val_losses = []
        # doing grid search for different lambda values
        for lbd in lambda_vals:
            # initialize the model
            model = RidgeRegression(theta, lbd)
            # initialize the optimizer
            optimizer = optim.SGD(model.parameters(), lr=args.lr)  # lr=1e-5

            # train the model
            for _ in range(args.num_epochs):  # num_epochs=1000
                model.train()
                y_pred = model(X_train)
                loss = model.loss(y_train, y_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # evaluate the model
            model.eval()
            y_val_pred = model(X_val)
            val_loss = model.loss(y_val, y_val_pred, val=True)
            val_losses.append(val_loss.item())
        #print the minimum lambda and the corresponding validation loss
        print(f"Minimum validation loss: {min(val_losses)}")
        print(f"Lambda value: {lambda_vals[val_losses.index(min(val_losses))]}")
        # plot the lambda vs validation loss
        plot_lambda_vs_val_loss(lambda_vals, val_losses)
    # selecting the model with hypergradients
    if args.model == "hg":

        model = RidgeRegressionHG(lambda_=args.lambda_init)  # lambda_=0.5
        val_losses = []
        lambdas = []
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  # lr=1e-3
        for _ in tqdm(range(args.num_epochs)):  # num_epochs=15001
            optimizer.zero_grad()
            # Forward pass
            theta_star = model(X_train, y_train)
            loss = model.loss(y_val, X_val, theta_star)
            val_losses.append(loss.item())
            lambdas.append(model.lambda_.item())
            # do backpropagation on the loss
            loss.backward()
            optimizer.step()

        # plot the loss vs iteration
        plot_loss_vs_iteration(val_losses)
        # plot the lambda vs iteration
        plot_lambda_vs_iteration(lambdas)


if __name__ == "__main__":
    main()
