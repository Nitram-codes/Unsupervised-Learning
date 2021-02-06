import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('PCAexampleData.txt', header = None)
col_names = ['x', 'y']
df.columns = col_names

# plot data
fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.scatter(df['x'], df['y'])

data = df.values
# m is the number of samples, n is the dimensionality of the data
m, n = data.shape

# rescale the data by subtracting the mean of each feature column from
# every element in the column
# eg. the mean of the column X is subtracted from every X[i]
xmean = np.mean(data[:,0])
ymean = np.mean(data[:,1])
xscaled = data[:,0] - xmean
yscaled = data[:,1] - ymean
scaled_data = np.stack((xscaled, yscaled), axis = 1)

# calculate the covariance matrix
Sigma = (1/(m - 1))*np.dot(scaled_data.T, scaled_data)
# calculate the eigenvalues and eigenvectors of the covariance matrix
eigenvals, eigenvecs = np.linalg.eig(Sigma)
# identify the primary components
# the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
# the primary component is the eigenvector corresponding to the largest eigenvalue
first_component = -1*eigenvecs[:,1]
second_component = -1*eigenvecs[:,0]
# check that these to eigenvectors are perpendicular
print('The dot product of the two primary components is equal to \
{}'.format(np.inner(first_component, second_component)))

# plot eigenvectors from the mean of the data points
ax.arrow(xmean, ymean, 3*first_component[0], 3*first_component[1], color = 'r', width = 0.01)
ax.arrow(xmean, ymean, 0.5*second_component[0], 0.5*second_component[1],
         color = 'r', width = 0.01)
plt.axis('scaled')

# project data onto 1 dimension using the 'first_component' eigenvector
projected_data = np.dot(data, first_component)
# plot projected data
# there needs to 2D data to use 2D plotting functions so create an array of zeros
# to plot the projected data along the x axis
zeros = np.zeros(projected_data.shape[0],)
ax2 = fig.add_subplot(122)
ax2.scatter(projected_data, zeros)
plt.show()