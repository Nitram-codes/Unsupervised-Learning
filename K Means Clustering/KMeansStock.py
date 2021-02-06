import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

df = pd.read_csv('stock_movements.csv', header = None)
df = df.T
# use the first row of df as the header
new_header = df.iloc[0]
# reassign df to the remainder of the df
df = df[1:]
# assign the column names
df.columns = new_header
# stock data
stocks = df.values.T

# create a normalizer to normalize the data
# normalizer normalizes each sample to the unit norm
# largest sample feature cannot be greater than 1
# all other features are then scaled accordingly
normalizer = Normalizer()
# create a K Means model with 10 clusters
kmeans = KMeans(n_clusters = 10)
# create a pipeline
pipeline = make_pipeline(normalizer, kmeans)
# fit pipeline to stock movements
pipeline.fit(stocks)
# predict the cluster labels for the companies
labels = pipeline.predict(stocks)
companies = df.columns
# create a dataframe aligning labels and companies
df1 = pd.DataFrame({'labels': labels, 'companies':companies})
# display df1 sorted by cluster labels
# this shows which companies have been clustered together
print(df1.sort_values('labels'))

# use the linkage function to perform hierarchial clustering
# and the dendrogram function to visualise the results
# when undertaking hierarchial clustering you have to use the normalize
# function instead or Normalizer
normalised_stocks = normalize(stocks)
# calculate the linkage
mergings = linkage(normalised_stocks, method = 'complete')
# plot the dendrogram
dendrogram(mergings, labels = companies, leaf_rotation = 90,
           leaf_font_size = 6)
plt.show()

