import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Load your dataset
data = pd.read_csv(r'D:\Current_Learning\TY_NOTES\ML\Practical\Exp_No_10\Experiment10.csv')

# Encode 'species' if it's categorical
if data['Species'].dtype == 'object':
    encoder = LabelEncoder()
    data['Species'] = encoder.fit_transform(data['Species'])

# Define features and target
X = data[['Weight', 'Width']]  
y = data['Species']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the SVM classification model
model = svm.SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

y_pred_labels = encoder.inverse_transform(y_pred)
print("SVM Predictions (Species):", y_pred_labels)
