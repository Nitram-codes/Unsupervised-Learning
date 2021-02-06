import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

df = pd.read_csv('seeds.csv')
col_names = ['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5',\
              'feature 6', 'feature 7', 'variety']
df.columns = col_names
samples = df[['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5',\
              'feature 6', 'feature 7']].values

# get the variety numbers into a list
variety_nums = []
for num in df['variety']:
    variety_nums.append(num)

# create a TSNE instance
# TSNE produces a representation of high-dimensional data in 2 dimensions
model = TSNE(learning_rate = 200)
# apply fit_transform to samples
tsne_features = model.fit_transform(samples)
# set x equal to the first column
x = tsne_features[:, 0]
# set y equal to the second column
y = tsne_features[:, 1]
plt.scatter(x, y, c = variety_nums)
plt.show()
