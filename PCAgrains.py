import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

df = pd.read_csv('grains.csv', header = None)
col_names = ['width', 'length']
df.columns = col_names

fig = plt.figure()
ax = fig.add_subplot(221)
ax.scatter(df['width'], df['length'])
ax.set_xlabel('width')
ax.set_ylabel('length')

# calculate the pearson correlation of width and length
correlation, pvalue = pearsonr(df['width'], df['length'])

# use PCA to decorrelate the data
# the mean of the data is shifted to zero and the data is rotated so there
# is no correlation
# create PCA instance
model = PCA()
# apply the fit_transform method to the data
pca_features = model.fit_transform(df.values)
x = pca_features[:, 0]
y = pca_features[:, 1]
ax2 = fig.add_subplot(222)
ax2.scatter(x, y)

# check the correlation
correlation2, pvalue2 = pearsonr(x, y)

# find and plot the first principal component of the data
# the vector aligned with the direction of primary variance in the data
# create a PCA instance
model2 = PCA(n_components = 1)
model2.fit(df.values)
# get the mean of the data samples
mean = model2.mean_
# get the first principal component
first_pc = model2.components_
# plot the vector
ax.arrow(mean[0], mean[1], first_pc[0][0], first_pc[0][1], color = 'r', width = 0.01)

# project the data onto 1 dimension using the principal component
projected_data = model2.transform(df.values)
# plot projected data
# there needs to 2D data to use 2D plotting functions so create an array of zeros
# to plot the projected data along the x axis
zeros = np.zeros(210,)
ax3 = fig.add_subplot(223)
ax3.scatter(projected_data, zeros)
plt.show()