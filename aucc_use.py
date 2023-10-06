from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from sklearn.cluster import KMeans

import numpy as np

from aucc import aucc

# As usage example, we use the same code employed for Silhouette in scikit-learn
# available at: https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

X, y = datasets.load_iris(return_X_y=True)
kmeans_model = KMeans(n_clusters=3, random_state=1, n_init='auto').fit(X)
labels = kmeans_model.labels_
print(aucc(X, labels, metric='euclidean'))