import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data

plt.scatter(X[:, 0],X[:, 1], s = 10)

km = KMeans(n_clusters = 5)
km.fit(X)

y_km = km.predict(X)
plt.scatter(X[:,0], X[:,1], c = y_km, s = 50, cmap = 'Accent')

centers = km.cluster_centers_

plt.scatter(centers[:,0], centers[:,1], c = 'red', s = 200, alpha = 0.5)
