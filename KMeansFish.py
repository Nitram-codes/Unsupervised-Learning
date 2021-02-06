import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('fish_data.csv')
col_names = ['species', 'feature1', 'feature2', 'feature3', 'feature4',\
             'feature5', 'feature6']
df.columns = col_names
samples = df[['feature1', 'feature2', 'feature3', 'feature4',\
             'feature5', 'feature6']].values
species = df['species'].values
# the data is unscaled and varies greatly in scale
# create scaler: standard score = (x - mean)/standard deviation
scaler = StandardScaler()
# create K Means instance
kmeans = KMeans(n_clusters = 4)
# create a pipeline
pipeline = make_pipeline(scaler, kmeans)
# fit the pipeline to the samples
pipeline.fit(samples)
# calculate the cluster labels
labels = pipeline.predict(samples)
# create a DataFrame of the labels and species for the crosstab function
df1 = pd.DataFrame({'labels': labels, 'species': species})
# use the crosstab function to count the number of times each fish species
# coincides with each cluster label
ct = pd.crosstab(df1['labels'], df1['species'])
