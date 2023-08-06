import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#Activation function which is sigmoid in our case...
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_crossentropy(y_true, y_pred):
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#--->The derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Loading  the training data using pandas dataframe
df = pd.read_csv('C:\\Users\\manoj\\OneDrive\\Desktop\\KDAG TASK\\KDAGTaskFinal\\ds2_train.csv')

X = df[['x_1', 'x_2']].values
y = df['y'].values

# Normalize the input data using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Transpose the data to match the dimensions of the network
X = X.T
y = y.T

# Define the number of neurons in each layer of the neural network
inputLayer_neurons = X.shape[0]  # number of features in the data set
hiddenLayer_neurons = 10 
outputLayer_neurons = 1

# Initialize random weights for input to hidden and hidden to output layers
weights_input_hidden = np.random.uniform(size=(inputLayer_neurons, hiddenLayer_neurons))
weights_hidden_output = np.random.uniform(size=(hiddenLayer_neurons, outputLayer_neurons))


epochs = 20000  # ---> Number of epochs
lr = 0.01  # Learning rate

losses = []


for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(weights_input_hidden.T, X)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(weights_hidden_output.T, hidden_layer_output)
    output = sigmoid(output_layer_input)

    # Calculate binary cross-entropy loss
    loss = np.mean(binary_crossentropy(y, output))
    losses.append(loss)

    # Backpropagation
    output_error = output - y
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = np.dot(weights_hidden_output, output_delta)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Update weights using gradient descent
    weights_hidden_output -= lr * np.dot(hidden_layer_output, output_delta.T)
    weights_input_hidden -= lr * np.dot(X, hidden_delta.T)

    if epoch % 100 == 0:
        print(f"Loss at epoch {epoch} is {loss:.5f}")

# Prediction and accuracy evaluation on the training data
hidden_layer_output = sigmoid(np.dot(weights_input_hidden.T, X))
output = sigmoid(np.dot(weights_hidden_output.T, hidden_layer_output))
y_pred = (output >= 0.5).astype(int).flatten()  
accuracy = accuracy_score(y_pred, y)
print("Final accuracy on training data:", accuracy)

# Loading the testing dataset
df_test = pd.read_csv('C:\\Users\\manoj\\OneDrive\\Desktop\\KDAG TASK\\KDAGTaskFinal\\ds2_test.csv')


X_test = df_test[['x_1', 'x_2']].values
y_test = df_test['y'].values


X_test = scaler.transform(X_test)

X_test = X_test.T
y_test = y_test.T

# Perform forward propagation using trained weights on the test data
hidden_layer_output = sigmoid(np.dot(weights_input_hidden.T, X_test))
output = sigmoid(np.dot(weights_hidden_output.T, hidden_layer_output))
y_pred = (output >= 0.5).astype(int).flatten()

# Calculate and print the testing accuracy
accuracy = accuracy_score(y_pred, y_test)
print("Testing accuracy:", accuracy)
