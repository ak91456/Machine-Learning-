import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
linear_x = pd.read_csv('LinearX.csv', header=None).values.flatten()
linear_y = pd.read_csv('LinearY.csv', header=None).values.flatten()

# Normalize the predictor
linear_x = (linear_x - np.mean(linear_x)) / np.std(linear_x)

# Prepare data
X = np.c_[np.ones((len(linear_x), 1)), linear_x]  # Add bias term
y = linear_y.reshape(-1, 1)

# Initialize parameters
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        gradients = (1 / m) * (X.T @ (X @ theta - y))
        theta -= learning_rate * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Perform gradient descent
learning_rate = 0.5
iterations = 50
theta = np.zeros((X.shape[1], 1))
optimal_theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

# Plot cost vs. iterations
plt.figure()
plt.plot(range(len(cost_history)), cost_history, label="Cost over Iterations", color='blue')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations (Learning Rate: 0.5)")
plt.legend()
plt.grid(True)
plt.show()

# Plot data and regression line
plt.figure()
plt.scatter(linear_x, linear_y, label='Data', color='blue')
plt.plot(linear_x, X @ optimal_theta, label='Regression Line', color='red')
plt.xlabel("Normalized Predictor (X)")
plt.ylabel("Response (y)")
plt.title("Data and Regression Line")
plt.legend()
plt.grid(True)
plt.show()

# Testing with different learning rates
learning_rates = [0.005, 0.5, 5]
plt.figure()
for lr in learning_rates:
    theta_temp = np.zeros((X.shape[1], 1))
    _, cost_history_lr = gradient_descent(X, y, theta_temp, lr, iterations)
    plt.plot(range(len(cost_history_lr)), cost_history_lr, label=f'lr={lr}')

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()

# Implementing Stochastic and Mini-Batch Gradient Descent
def stochastic_gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        for j in range(m):
            random_index = np.random.randint(m)
            X_i = X[random_index:random_index+1]
            y_i = y[random_index:random_index+1]
            gradients = X_i.T @ (X_i @ theta - y_i)
            theta -= learning_rate * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history

def mini_batch_gradient_descent(X, y, theta, learning_rate, iterations, batch_size):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for j in range(0, m, batch_size):
            X_i = X_shuffled[j:j+batch_size]
            y_i = y_shuffled[j:j+batch_size]
            gradients = (1 / len(y_i)) * X_i.T @ (X_i @ theta - y_i)
            theta -= learning_rate * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history

# Parameters
theta_sgd = np.zeros((X.shape[1], 1))
theta_mbgd = np.zeros((X.shape[1], 1))
batch_size = 10

# Perform Gradient Descent Methods
_, cost_history_sgd = stochastic_gradient_descent(X, y, theta_sgd, learning_rate=0.1, iterations=50)
_, cost_history_mbgd = mini_batch_gradient_descent(X, y, theta_mbgd, learning_rate=0.1, iterations=50, batch_size=batch_size)

# Plotting Cost Functions
plt.figure()
plt.plot(range(len(cost_history_sgd)), cost_history_sgd, label="Stochastic GD", linestyle="--", color='green')
plt.plot(range(len(cost_history_mbgd)), cost_history_mbgd, label="Mini-Batch GD", linestyle="-.", color='orange')
plt.plot(range(len(cost_history)), cost_history, label="Batch GD", linestyle="-", color='blue')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations for Different Gradient Descent Methods")
plt.legend()
plt.grid(True)
plt.savefig('gradient_descent_comparison.png')
plt.show()
