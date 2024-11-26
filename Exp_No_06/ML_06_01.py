# Import libraries
import sys
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r"D:\Current_Learning\TY_NOTES\ML\Practical\Exp_No_06\Experiment6.csv")

# Map categorical variables to numerical values
nationality_map = {'UK': 1, 'USA': 0, 'N': 2}
df['Nationality'] = df['Nationality'].map(nationality_map)
go_map = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(go_map)

# Define all possible feature pairs to explore
feature_combinations = [
    ['Age', 'Experience'],
    ['Age', 'Rank'],
    ['Age', 'Nationality'],
    ['Experience', 'Rank'],
    ['Experience', 'Nationality'],
    ['Rank', 'Nationality']
]

# Iterate through each feature combination and plot the tree
for i, features in enumerate(feature_combinations):
    X = df[features]
    y = df['Go']
    
    # Initialize and fit the Decision Tree Classifier
    dtree = DecisionTreeClassifier()
    dtree.fit(X, y)
    
    # Plot the decision tree
    plt.figure(figsize=(10, 8))
    tree.plot_tree(dtree, feature_names=features, class_names=['NO', 'YES'], filled=True)
    
    # Save the plot with a unique name for each feature combination
    plt.savefig(f"decision_tree_{i+1}.png")
    plt.close()

print("Decision trees for different feature combinations have been saved.")
