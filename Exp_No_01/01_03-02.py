#Line Chart


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Create a line chart for sepal length
df['sepal.length'].plot(kind='line', marker='o')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Sepal Length')
plt.title('Sepal Length Over Index')

# Display the plot
plt.show()
