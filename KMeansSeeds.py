import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('seeds.csv')
col_names = ['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5',\
              'feature 6', 'feature 7', 'variety']
df.columns = col_names
# determine the optimal number of clusters
k_vals = range(1, 10)
inertias = []
fig = plt.figure()
ax = fig.add_subplot(111)

for k in k_vals:
    # instantiate model with k clusters
    model = KMeans(n_clusters = k)
    # fit model to data
    model.fit(df)
    # append the inertia measure of cluster grouping to the list 'inertias'
    inertias.append(model.inertia_)

# plot k vs cluster intertia to determine the optimal k value for the clustering
# this is where the decrease in inertia starts to level off (k = 3 here)
ax.plot(k_vals, inertias)
ax.set_xlabel('Number of clusters, K')
ax.set_ylabel('Cluster inertia')
plt.show()