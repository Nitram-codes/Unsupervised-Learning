import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# import the data from file and split into x and y arrays
infile = open('KMeans_data.txt')
x_data = []
y_data = []
for line in infile:
    data = line.split()
    x_data.append(float(data[0]))
    y_data.append(float(data[1]))
# convert to numpy arrays
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
# combine x and y arrays for later use
all_data = np.stack((x_data, y_data), axis = 1)

# plot the data to view point distribution
fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(x_data, y_data)
ax.set_xlabel('Feature x')
ax.set_ylabel('Feature y')

# from the plot it is evident that there are 3 clusters of points
# instantiate a KMeans cluster instance with 3 clusters
model = KMeans(n_clusters = 3)
# fit model to data points
model.fit(all_data)
# fefine new points
new_points = np.array([-1.2, -1, 1.5, 0.5, 2, 0.3, -0.12, -0.8]).reshape(4, 2)
# predict cluster labels of new points
labels = model.predict(new_points)
# find the cluster centroids
centroids = model.cluster_centers_
# x coordinates of centroids
centroids_x = centroids[:,0]
# y coordinates of centroids
centroids_y = centroids[:,1]
ax.scatter(centroids_x, centroids_y, marker = 'D', s = 50)

# identify the optimum number of clusters for model
k_vals = range(1, 10)
inertias = []
ax2 = fig.add_subplot(212)

for k in k_vals:
    # instantiate model with k clusters
    model2 = KMeans(n_clusters = k)
    # fit model to data
    model2.fit(all_data)
    # append the inertia measure of cluster grouping to the list 'inertias'
    inertias.append(model2.inertia_)

ax2.plot(k_vals, inertias)
ax2.set_xlabel('Number of clusters, K')
ax2.set_ylabel('Cluster inertia')
plt.show()
