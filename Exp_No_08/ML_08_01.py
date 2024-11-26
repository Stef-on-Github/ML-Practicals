from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
glass = pd.read_csv(r"D:\Current_Learning\TY_NOTES\ML\Practical\Exp_No_08\Experiment8.csv")

# Print dataset information
print("Shape of dataset:", glass.shape)
print("Null values per column:\n", glass.isnull().sum())

# Plot distribution of Glass Types
plt.figure(figsize=(8, 6))
sns.countplot(x='Type', data=glass, color='red')
plt.title("Distribution of Glass Types")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

# Initialize Gaussian Naive Bayes model
nb = GaussianNB()

# Separate features and target variable
X = glass.drop(columns=['Type'])
y = glass['Type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Train the model on the training data
nb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb.predict(X_test)
print("Predicted values:", y_pred)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
