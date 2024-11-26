#Scatter Plot

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Create a scatter plot for sepal length vs. sepal width
df.plot(kind='scatter', x='sepal.length', y='sepal.width')

# Add labels and title
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')

# Display the plot
plt.show()
