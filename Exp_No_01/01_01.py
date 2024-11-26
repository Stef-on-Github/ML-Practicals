import pandas as pd  # Import the pandas library for data manipulation
import numpy as np   # Import the numpy library (though it's not used in this script)

# Define the path to the CSV file
file_path = "Iris.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print("First few rows:")
print(df.head())

# Display the last few rows of the DataFrame
print("\nLast few rows:")
print(df.tail())

# Display the shape of the DataFrame (number of rows and columns)
print("\nShape of the DataFrame:")
print(df.shape)

# Display basic information about the DataFrame, including column data types and non-null counts
print("\nInformation about the DataFrame:")
print(df.info())

# Display the total number of elements in the DataFrame (rows * columns)
print("\nTotal number of elements in the DataFrame:")
print(df.size)

# Display information about missing values in the DataFrame (this method needs parentheses)
print("\nMissing values in the DataFrame:")
print(df.isna())
print(df.isna().sum())

# Display descriptive statistics of the DataFrame, such as mean, standard deviation, and quartiles
print("\nDescriptive statistics of the DataFrame:")
print(df.describe())

# Display the number of unique values for each column in the DataFrame
print("\nNumber of unique values per column:")
print(df.nunique())
