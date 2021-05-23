import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import random
import sys

df = pd.read_csv('/home/runner/kaggle/titanic/processed_titanic_data.csv')

features_to_use = ["Sex", "Pclass", "Fare", "Age", "SibSp"]
final_df = df[features_to_use + ['Survived']]

df = df[features_to_use]

k_values = [n for n in range(1, 26)]


for f in df.columns:
    df[f] = (df[f] - df[f].min()) / (df[f].max() - df[f].min())


distortions = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df)
    distortions.append(kmeans.inertia_)

plt.style.use('bmh')
plt.plot(k_values, distortions)
plt.xticks(range(1, 26))
plt.xlabel('k')
plt.ylabel('sum squared distances from clustered center')
plt.title("K-Means Clustering on Titanic Dataset")
plt.savefig('titanic/elbow_method_on_titanic_dataset.png')


kmeans = KMeans(n_clusters=4, random_state=0).fit(df)
final_df['cluster'] = kmeans.labels_
final_df['count'] = final_df['cluster'].apply(lambda entry: sum([1 for label in kmeans.labels_ if label == entry]))

print(final_df.groupby(['cluster']).mean())