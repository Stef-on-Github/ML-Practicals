import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
file = "diabetes.csv"
df = pd.read_csv(file)

# Separate features and target variable
data = df.values
X, y = data[:, :-1], data[:, -1]
print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(f"Training feature shape: {X_train.shape}, Testing feature shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}, Testing target shape: {y_test.shape}")

# Initialize and fit the KNN model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Make predictions
y_predict = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_predict)
print("Dataframe:\n", df.head())  # Print only the first few rows for clarity
print("Accuracy: {:.2f}%".format(accuracy * 100))
