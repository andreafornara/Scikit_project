# %%
"""
## Learn with us: www.zerotodeeplearning.com

Copyright © 2021: Zero to Deep Learning ® Catalit LLC.
"""

# %%
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
"""
Documentation links:

- [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)
- [Numpy](https://docs.scipy.org/doc/)
- [Pandas](https://pandas.pydata.org/docs/getting_started/index.html)
- [Pandas Cheatsheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Matplotlib](https://matplotlib.org/)
- [Matplotlib Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)
- [Scikit-learn Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf)
- [Scikit-learn Flow Chart](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
"""

# %%
"""
# Clustering with Scikit Learn
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
# %%
url = "https://raw.githubusercontent.com/zerotodeeplearning/ztdl-masterclasses/master/data/"
# %%
df = pd.read_csv(url + 'iris.csv')
# %%
df.head()
# %%
df.plot.scatter(x='sepal_length', y='petal_length', title='Iris Flowers')
# %%
X = df.drop('species', axis=1).values

# %%
# We start with the easiest clustering algorithm, KMeans
# From the picture we see two clusters, so we input 2
model = KMeans(2)
model.fit(X)

# %%
centers = model.cluster_centers_
# The centers are 4D because each flower has 4 features (+species that we dropped)
centers

# %%
plt.scatter(df['sepal_length'], df['petal_length'], c=model.labels_)
plt.scatter(centers[:, 0], centers[:, 2], marker='o', c='r', s=100)
plt.xlabel('sepal_length')
plt.ylabel('petal_length')

# %%
# Are we sure that this is the correct number?
# Let's use the elbow method to find the optimal k (squared distance from the cluster centers)
ks = range(1, 11)
inertia = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(X)
    # Notice that model.inertia is defined with an underscore at the end:
    # all the methods that end with an underscore are "sklearn" conventions
    # showing that the method can only be called after the fit has been run
    inertia.append(model.inertia_)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ks, inertia, marker='o')
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method')
plt.show()
# So 2 or 3 are good candidates for the optimal k (where the elbow is)

# %%
# We can use another score, the silhouette
ks = range(2, 11)
silhouette_scores = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(X)
    silhouette_scores.append(silhouette_score(X, model.labels_))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ks, silhouette_scores, marker='o')
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Method')
plt.show()
# The best result is at k = 2 (maximum of the silhouette score)

# %%
"""
### Exercise 3

- Switch clustering method to `DBSCAN` and fit the model for various values of `eps`
- calculate the `silhouette_score` for each value
- determine how many clusters were found for each value
- Bonus points if you plot the `silhouette_score` as a function of `eps`
"""

# %%
def plot_clusters(model):
    plt.scatter(df['sepal_length'], df['petal_length'], c=model.labels_)
    plt.xlabel('sepal_length')
    plt.ylabel('petal_length')
    plt.title('DBSCAN Clustering')
plt.figure(figsize=(15,15))
eps_values = [0.25,0.3,0.4,0.5,0.75,1,1.25,1.5]
silhouette_scores = []
n_clusters = []
i = 1
for eps in eps_values:
  model = DBSCAN(eps=eps)
  model.fit(X)
  score = silhouette_score(X,model.labels_)
  silhouette_scores.append(score)
  plt.subplot(3,3,i)
  plot_clusters(model)
  plt.title(f"eps={eps}, n_clusters={len(set(model.labels_))}, silhouette={score:.2f}")
  i += 1
plt.subplot(3,3,i)
plt.plot(eps_values, silhouette_scores, marker='o')
plt.xlabel('Epsilon (eps)')
plt.ylabel('Silhouette Score')
plt.title('DBSCAN Silhouette Score')
plt.show()

# From the silhouette score we can see that, after eps = 1, we get a stable maximum in silhouette
# This confirms that we probably have two clusters