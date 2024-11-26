import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Salary_dataset.csv')
X = dataset.iloc[:, 0:1].values  # Features (Years of Experience)
y = dataset.iloc[:, 1].values      # Target variable (Salary)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fitting the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Print shapes for debugging
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Visualizing the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Using training data for the line
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Displaying predictions
for pred in y_pred:
    print("Predicted Salary:", pred)
