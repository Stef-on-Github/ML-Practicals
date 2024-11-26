import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Find and print the minimum value in the array
print("Minimum value in the array:", np.min(arr))

# Find and print the maximum value in the array
print("Maximum value in the array:", np.max(arr))

# Compute and print the mean (average) value of the array
print("Mean (average) value of the array:", np.mean(arr))

# Compute and print the standard deviation of the array
print("Standard deviation of the array:", np.std(arr))

# Compute and print the median value of the array
print("Median value of the array:", np.median(arr))

# Compute and print the 50th percentile (median) value of the array
print("50th percentile (median) value of the array:", np.percentile(arr, 50))

# Generate an array of 5 evenly spaced numbers between 0 and 10
arr = np.linspace(0, 10, 5)
print("Array of 5 evenly spaced numbers between 0 and 10:", arr)

# Create a 2D NumPy array
arr = np.array([[1, 2], [3, 4]])

# Print the shape of the array (number of rows and columns)
print("Shape of the 2D array:", arr.shape)

# Create a 1D NumPy array and reshape it into a 2D array with 2 rows and 3 columns
arr = np.array([1, 2, 3, 4, 5, 6])
reshaped_arr = arr.reshape((2, 3))
print("Reshaped 2D array with 2 rows and 3 columns:")
print(reshaped_arr)

# Create two NumPy arrays and copy the values from the source array to the destination array
dest = np.array([0, 0, 0])
src = np.array([1, 2, 3])
np.copyto(dest, src)
print("Destination array after copying values from the source array:")
print(dest)

# Print the transpose of the 2D array (swap rows with columns)
print("Transpose of the 2D array (swap rows with columns):")
print(arr.T)

# Create two NumPy arrays and stack them along a new axis (row-wise)
arr1 = np.array([1, 2])
arr2 = np.array([3, 4])
stacked = np.stack((arr1, arr2), axis=0)
print("Arrays stacked along a new axis (row-wise):")
print(stacked)

# Vertically stack the two arrays (row-wise)
vstacked = np.vstack((arr1, arr2))
print("Arrays stacked vertically (row-wise):")
print(vstacked)

# Horizontally stack the two arrays (column-wise)
hstacked = np.hstack((arr1, arr2))
print("Arrays stacked horizontally (column-wise):")
print(hstacked)
