#Bar Chart


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Creating a bar chart for the count of each variety
df['variety'].value_counts().plot(kind='bar')

# Adding labels and title
plt.xlabel('Variety')
plt.ylabel('Count')
plt.title('Count of Each Iris Variety')

# Displaying the plot
plt.show()
