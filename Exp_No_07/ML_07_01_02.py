import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r"C:\Machiine_Learning\Practical\Experiment7.csv")

# Take only the first 100 records
data = data.head(50)

# Encode categorical variables
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

# Define features and target variable
X = data.drop(columns=['Survived'])
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=10)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)

# Plot and save the first three decision trees from the forest
n_trees = 3  # Number of trees to plot

for i in range(n_trees):
    plt.figure(figsize=(20, 15))  # Increase the figure size to accommodate the text
    plot_tree(rf_model.estimators_[i], 
              filled=True, 
              feature_names=X.columns, 
              class_names=['0', '1'], 
              rounded=True, 
              proportion=True,
              fontsize=12)  # Set fontsize to make the text inside the boxes more readable
    plt.title(f"Decision Tree {i + 1}", fontsize=20)
    # Save the tree as an image
    plt.savefig(f"C:\\Machine Learning\\Random Forest Algorithm\\decision_tree_{i+1}.png", bbox_inches='tight')
    plt.close()  # Close the plot after saving to prevent overlapping in the next iteration

print("Decision trees saved as images.")
