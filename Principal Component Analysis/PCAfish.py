from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('fish_data.csv', header = None)
col_names = ['species', 'feature1', 'feature2', 'feature3', 'feature4',\
             'feature5', 'feature6']
df.columns = col_names

# create a scaler, as feature1 has much larger values than the other features
scaler = StandardScaler()
# create a PCA instance
pca = PCA()
# create pipeline
pipeline = make_pipeline(scaler, pca)
# fit the pipeline to the data
data = df[['feature1', 'feature2', 'feature3', 'feature4',\
             'feature5', 'feature6']].values
pipeline.fit(data)

# plot the variances
# n_components is the estimated dimensionality of the data
features = range(pca.n_components_)
fig = plt.figure()
ax = fig.add_subplot(211)
# explained_variance calculates the variance in each feature
ax.bar(features, pca.explained_variance_)
ax.set_xlabel('PCA feature')
ax.set_ylabel('Variance')
ax.set_xticks(features)


# now we will project the data to 2 dimensions as suggested by the bar plot
# create a pca method which will use two principal components
pca2 = PCA(n_components = 2)
# scale the data, as feature1 has much larger values than other features
scaled_data = scaler.fit_transform(data)
# fit the data
pca2.fit(scaled_data)
# identify the principal components (eienvectors of covariance matrix)
principal_components = pca2.components_
# project 'scaled_data' onto the principal components
projected_data2 = pca2.transform(scaled_data)
# plot the projected data
ax2 = fig.add_subplot(212)
ax2.scatter(projected_data2[:,0], -projected_data2[:,1])
plt.show()