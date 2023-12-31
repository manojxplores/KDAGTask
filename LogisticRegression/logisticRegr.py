import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine

#loading the data using pandas
data = pd.read_csv("DATAset/ds2_train.csv")
x_train = data[['x_1', 'x_2']].values
y_train = data['y'].values

#Define the activation function which is sigmoid in our case
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function for logistic regression with L2 regularization
def compute_cost(x, y, w, b, lambda_reg):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        z = np.dot(x[i], w) + b
        f_x = sigmoid(z)
        cost += -y[i] * np.log(f_x) - (1 - y[i]) * np.log(1 - f_x)

    # Add L2 regularization term
    l2_reg = (lambda_reg / (2 * m)) * np.sum(w**2)
    cost = (cost / m) + l2_reg
    return cost

# Define the gradient function for gradient descent with L2 regularization
def gradient(x, y, w, b, lambda_reg):
    m = x.shape[0]
    n = x.shape[1]
    db = 0
    dw = np.zeros(x.shape[1])

    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_x_i = sigmoid(z_i)

        for j in range(n):
            dw[j] += (f_x_i - y[i]) * x[i, j]
            db += (f_x_i - y[i])

    # Add L2 regularization terms to the gradients
    dw = (dw / m) + ((lambda_reg / m) * w)
    db = db / m

    return dw, db

# Updating the parameters using gradient-descent algorithm
def gradient_descent(x, y, w, b, alpha, num_iter, lambda_reg):
    costs = []  # List to store the loss at each iteration

    for i in range(num_iter):
        dw, db = gradient(x, y, w, b, lambda_reg)

        w = w - alpha * dw
        b = b - alpha * db

        # Compute the loss and store it in the 'costs' list
        cost = compute_cost(x, y, w, b, lambda_reg)
        costs.append(cost)

        # Print the loss after each iteration
        print(f"Iteration {i+1}/{num_iter} - Loss: {cost}")

    return w, b, costs

# Initialize the weights and bias for training
w_init = np.zeros(x_train.shape[1])
b_init = 0

# Set hyperparameters for training
num_iterations = 5000  # Number of iterations
alpha = 0.01
lambda_reg = 0.1  # L2 regularization strength

# Call the gradient_descent function and get the updated weights, bias, and loss
w_train, b_train, loss_history = gradient_descent(x_train, y_train, w_init, b_init, alpha, num_iterations, lambda_reg)


# Define the predict function to make predictions on new data
def predict(x, w, b):
    m = x.shape[0]
    y_est = np.zeros(m)
    for i in range(m):
        z_i = np.dot(x[i], w) + b
        y_est[i] = sigmoid(z_i)
        if y_est[i] >= 0.5:
            y_est[i] = 1
        else:
            y_est[i] = 0
    return y_est

# Load the testing dataset
df_test = pd.read_csv('DATAset/ds2_test.csv')

# Extract features (x_test) and target variable (y_test) from the test data
x_test = df_test[['x_1', 'x_2']].values
y_test = df_test['y'].values

# Make predictions on the test data using the trained weights and bias
y_est = predict(x_test, w_train, b_train)

# Convert the predicted probabilities to binary predictions
y_pred = (y_est >= 0.5).astype(int)

# Calculate and print the testing accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Testing accuracy:", accuracy)
