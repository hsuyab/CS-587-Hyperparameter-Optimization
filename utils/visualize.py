import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Plotting functions for Ridge Regression
def plot_lambda_vs_val_loss(lambda_vals, val_losses):
    plt.plot(lambda_vals, val_losses)
    plt.xlabel('Lambda')
    plt.ylabel('Validation Loss')
    plt.title('Lambda vs Validation Loss')
    plt.show()
    plt.savefig('img/lambda_vs_val_loss_ridge_regression.png')

# Plotting functions for Ridge Regression with Hypergradients
def plot_loss_vs_iteration(val_losses):
    plt.figure()
    plt.plot(val_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss vs. Iteration (Ridge Regression)')
    plt.show()
    plt.savefig('img/val_loss_vs_iterations_ridge_regression.png')

# Plotting functions for Ridge Regression with lambda vs hypergradient iteration
def plot_lambda_vs_iteration(lambdas, model_type):
    plt.figure()
    plt.plot(lambdas)
    plt.xlabel('Iteration')
    plt.ylabel('Lambda')
    plt.title(f'Lambda vs. Iteration (Ridge Regression)')
    plt.show()
    plt.savefig('img/lambda_vs_iterations_ridge_regression.png')