[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Ridge Regression for Diabetes Prediction

This project implements Ridge Regression models for predicting diabetes using a specific dataset.

## Setup and Installation

To install the dependencies for the project, run the following command:

```bash
conda create -n cshw5 python=3.11.9
conda activate cshw5
pip install -r requirements.txt
```
OR 

```bash
conda env create -f environment.yml
conda activate cshw5
```

## Dependencies

The project requires the following major dependencies:

- Python 3.11.9
- NumPy
- Matplotlib
- scikit-learn
- PyTorch 2.1.2



# Experiments

The project includes two Ridge Regression models: 
- a basic Ridge Regression model
- a hyperparameter-tuned Ridge Regression model. 

## Basic Ridge Regression Model

The basic Ridge Regression model is implemented in `models/ridge_regression.py`. The model is trained on the diabetes dataset using standard Ridge Regression. The model is trained with the following hyperparameters:

- Learning rate: 1e-5
- Number of epochs: 1000
- Initial lambda value: 0.5

To run the basic Ridge Regression model, use the following command:

```bash
python main.py --model='normal' --lr=1e-5 --num_epochs=1000 --lambda_init=0.5
```
- The output image files will be saved in the `img/` directory as `img lambda_vs_val_loss_ridge_regression.png`



## Hyperparameter-Tuned Ridge Regression Model

The hyperparameter-tuned Ridge Regression model is implemented in `models/ridge_regression_hg.py`. The model uses hypergradient optimization to find the optimal lambda value for Ridge Regression. The model is trained with the following hyperparameters:

- Learning rate: 1e-3
- Number of epochs: 15001
- Initial lambda value: 0.5

To run the hypergradient based Ridge Regression model, use the following command:

```bash
python main.py --model='hg' --lr='1e-3' --num_epochs=15001 --lambda_init=0.5
```

- The output image files will be saved in the `img/` directory as `img/val_loss_vs_iterations_ridge_regression.png` and `img/lambda_vs_iterations_ridge_regression.png`



## Project Structure

The project has the following structure:

- `datasets/`: Contains the dataset loading and preprocessing code.
  - `diabetes_dataset.py`: Loads and preprocesses the diabetes dataset.

- `models/`: Contains the implementation of the Ridge Regression models.
  - `ridge_regression.py`: Implements the basic Ridge Regression model.
  - `ridge_regression_hg.py`: Implements a hyperparameter-tuned Ridge Regression model.

- `utils/`: Contains utility functions used in the project.
  - `visualize.py`: Provides visualization functions for plotting model performance.

- `tests/`: Contains unit tests for the models.
  - `test_models.py`: Defines unit tests for the Ridge Regression models.

- `img/`: Contains generated plots and visualizations.
  - `lambda_vs_iterations_ridge_regression.png`: Plot of lambda value vs. number of iterations for Ridge Regression.
  - `lambda_vs_val_loss_ridge_regression.png`: Plot of lambda value vs. validation loss for Ridge Regression.
  - `val_loss_vs_iterations_ridge_regression.png`: Plot of validation loss vs. number of iterations for Ridge Regression.

- `main.py`: The main script to run the Ridge Regression models and generate visualizations.

## Dataset

The project uses a diabetes dataset for training and evaluating the Ridge Regression models. The dataset is loaded and preprocessed using the code in `datasets/diabetes_dataset.py`.

## Models

The project includes two Ridge Regression models:

1. Basic Ridge Regression: Implemented in `models/ridge_regression.py`, this model performs standard Ridge Regression on the diabetes dataset.

2. Hyperparameter-tuned Ridge Regression: Implemented in `models/ridge_regression_hg.py`, this model uses hyperparameter tuning to optimize the Ridge Regression performance.

## Visualizations

The project generates visualizations to analyze the performance of the Ridge Regression models. The visualizations are created using the functions in `utils/visualize.py` and are saved in the `img/` directory.

The generated visualizations include:

- Lambda value vs. number of iterations for Ridge Regression
- Lambda value vs. validation loss for Ridge Regression
- Validation loss vs. number of iterations for Ridge Regression

## Usage

To run the project, execute the `main.py` script with the appropriate command-line arguments.

To recreate the Ridge Regression model without hypergradient optimization, use the following command:

```bash
python main.py --model='normal' --lr='1e-5' --num_epochs=1000 --lambda_init=0.5
```

To recreate the Ridge Regression model with hypergradient optimization, use the following command:

```bash
python main.py --model='hg' --lr='1e-3' --num_epochs=15001 --lambda_init=0.5
```


# Code Style

This project adheres to the PEP 8 style guidelines and is formatted using the black code formatter. All Python files have been automatically formatted with black to ensure a consistent and readable code style throughout the project.

To format the code using black, run the following command:
```bash
black .