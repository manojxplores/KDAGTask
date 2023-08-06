import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Load the training data
data = pd.read_csv('C:\\Users\\manoj\\OneDrive\\Desktop\\KDAG TASK\\KDAGTaskFinal\\ds1_train.csv')
X = data[['x_1', 'x_2']]
Y = data['y']

# Initialize the MLPClassifier with default parameters
model_neural_network = MLPClassifier(solver='lbfgs', alpha=0.01, max_iter=5000, hidden_layer_sizes=(5,), random_state=1)

# Define the parameters to search for the best combination
parameters = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'hidden_layer_sizes': [(3,), (4,), (5,)]
}

# Perform grid search with cross-validation
model_neural_network = GridSearchCV(model_neural_network, parameters, cv=5, scoring='accuracy')
model_neural_network.fit(X, Y)

# Load the test data
data_test = pd.read_csv('C:\\Users\\manoj\\OneDrive\\Desktop\\KDAG TASK\\KDAGTaskFinal\\ds1_test.csv')
X_test = data_test[['x_1', 'x_2']]
Y_test = data_test['y']

# Print the best parameters found during grid search
print(f'Best parameters: {model_neural_network.best_params_}')

# Evaluate the model's performance on the training data
train_accuracy = model_neural_network.best_estimator_.score(X, Y)
print(f'Training accuracy: {train_accuracy:.4f}')

# Evaluate the model's performance on the test data
test_accuracy = model_neural_network.best_estimator_.score(X_test, Y_test)
print(f'Test accuracy: {test_accuracy:.4f}')
