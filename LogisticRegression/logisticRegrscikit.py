import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('C:\\Users\\manoj\\OneDrive\\Desktop\\KDAG TASK\\KDAGTaskFinal\\ds1_train.csv')
# --->Loading the training data

# Separate features (X) and target variable (Y) from the training data
X = data[['x_1', 'x_2']]
Y = data['y']

# Creating a Logistic Regression model using scikit-learn
logistic_model = LogisticRegression()

#---> Define a paramter grid
parameters = {'C': [0.01, 0.1, 1, 10, 100],  
              'max_iter': [1000, 2500, 5000],  
              'solver': ['lbfgs', 'newton-cg']  
            }

# Create a GridSearchCV object to find the best hyperparameters 
logistic_model = GridSearchCV(logistic_model, parameters, cv=3, verbose=True)

# Fit the model to the training data
logistic_model.fit(X, Y)

# Load the test data
data_test = pd.read_csv('C:\\Users\\manoj\\OneDrive\\Desktop\\KDAG TASK\\KDAGTaskFinal\\ds1_test.csv')

X_test = data_test[['x_1', 'x_2']]
Y_test = data_test['y']

# Print the best hyperparameters found during grid search
print(f'Best parameters: {logistic_model.best_params_}')

# Evaluate the model's performance on the training data
train_accuracy = logistic_model.best_estimator_.score(X, Y)
print(f'Training accuracy: {train_accuracy:.4f}')

# Evaluate the model's performance on the test data
test_accuracy = logistic_model.best_estimator_.score(X_test, Y_test)
print(f'Test accuracy: {test_accuracy:.4f}')
