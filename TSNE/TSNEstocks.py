import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

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
normalised_stocks = normalize(stocks)
companies = []
# get the company names from df and add to list
for company in new_header:
    companies.append(company)

# create a TSNE instance
model = TSNE(learning_rate = 50)
# apply fit_transform to normalised_stocks
tsne_features = model.fit_transform(normalised_stocks)
# set x equal to the first column
x = tsne_features[:, 0]
# set y equal to the second column
y = tsne_features[:, 1]
# Scatter plot
plt.scatter(x, y, alpha = 0.5)
# Annotate the points
for x, y, company in zip(x, y, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()