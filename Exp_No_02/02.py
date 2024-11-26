import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
x_true = [1, 0, 2, 2, 1, 0]
x_pred = [2, 0, 2, 2, 0, 0]
print(confusion_matrix(x_true, x_pred))

y_true = ["cat", "ant", "cat", "ant", "cat", "bird"]
y_pred = ["cat", "ant", "ant", "cat", "ant", "cat"]
print(confusion_matrix(y_true, y_pred, labels = ["ant", "cat", "bird"]))


from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Loading the breast cancer data set
diabetes_data = load_breast_cancer()

# Creating independent and dependent variables
X = diabetes_data.data
y = diabetes_data.target

# Splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=24)
print(f"Train Data: {X_train.shape}, {y_train.shape}")
print(f"Test Data: {X_test.shape}, {y_test.shape}")

# Training a binary classifier using Random Forest Algorithm with default hyperparameters
classifier = RandomForestClassifier(random_state=18)
classifier.fit(X_train, y_train)

# Here X_test, y_test are the test data points
predictions = classifier.predict(X_test)

#Importing all necessary libraries
from sklearn.metrics import accuracy_score

# Calculating the accuracy of classifier
print(f"Accuracy of the classifier is: {accuracy_score(y_test, predictions)}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Compute and print the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=diabetes_data.target_names)
disp.plot()
plt.show()