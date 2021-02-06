import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('fish_data.csv', header = None)
# name the columns
col_names = ['species', 'feature1', 'feature2', 'feature3', 'feature4',\
             'feature5', 'feature6']
df.columns = col_names

# rescale the data by subtracting the mean from each column from each
# element in the column and then dividing each element by the standard
# deviation of the column
# x_i = (x_i - mean)/standard_deviation
# we have to divide by the standard deviation as the 'feature1' data
# is so much larger than all the other columns
for column in range(1, 7):
    df.iloc[:, column] = (df.iloc[:, column] - df.iloc[:,column].mean())/df.iloc[:,column].std()

data = df[['feature1', 'feature2', 'feature3', 'feature4',\
             'feature5', 'feature6']].values

# m is the number of samples, n is the dimensionality of the data
m, n = data.shape
# calculate the covariance matrix
Sigma = (1/(m - 1))*np.dot(data.T, data)
# calculate the eigenvalues and eigenvectors of the covariance matrix
eigenvals, eigenvecs = np.linalg.eig(Sigma)
# we want to identify the first two primary components for a projection
# to 2 dimensions
# the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
# the primary component is the eigenvector corresponding to the largest eigenvalue
first_component = eigenvecs[:,0]
second_component = eigenvecs[:,1]
U = np.stack((first_component, second_component), axis = 1)

# project the data onto 2 dimensions using U
projected_data = np.dot(data, U)
# plot the projected data in 2 dimensions
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(projected_data[:,0], projected_data[:, 1])
plt.show()