# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import time
# %%
"""
### Load data
"""

# %%
url = "https://raw.githubusercontent.com/zerotodeeplearning/ztdl-masterclasses/master/data/"

# %%
#We take a look at the file
df = pd.read_csv(url + 'geoloc_elev.csv')
df.head()
# %%
# We can see that 'source' is a categorical variable with 3 classes
# It probably is S = Survey, C = Civilian Data, Q = Questionable
df['source'].value_counts()

# %%
# In target instead we have 2 classes, one is 0 and the other is 1
# This is indeed a binary classification problem
df['target'].value_counts()
# %%
# Ok, by looking at the pairplots we can see that the classes are somewhat separable
# if we use the lat-lon space, whereas elev-lat or elev-lon do not provide
# a good way to separate those two classes
sns.pairplot(df, hue='target')
# %%
y = df['target']
# This is useful, we drop from the df the target column
raw_features = df.drop('target', axis=1)
# This is smart: it converts each categorical variable in a different column
# Each column will be one-hot encoded false/true statement 
X = pd.get_dummies(raw_features)
X.head()

#So now our X has the three columns for C,Q,S and three columns of 'data'
# %%
# We split. the data in train/test as usual
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size = 0.2, random_state=0)

# %%
# And now we use a decision tree classifier!
#
model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %%
#Did it work? Let's look at the confusion matrix
cm = confusion_matrix(y_test, y_pred)

pd.DataFrame(cm,
             index=["Miss", "Hit"],
             columns=['pred_Miss', 'pred_Hit'])

# %%
# The model is not doing so great with a depth of 3.
# We can investigate what is going on by doing a plot of the 
# decision boundary
print(classification_report(y_test, y_pred))
# %%
def plot_decision_boundary(model):
  hticks = np.linspace(-2, 2, 101)
  vticks = np.linspace(-2, 2, 101)

  aa, bb = np.meshgrid(hticks, vticks)
  a_flat = aa.ravel()
  b_flat = bb.ravel()
  N = len(a_flat)
  # The model expects 6 entries but we want to plot only 2
  # So we provide 4 zeros
  zeros = np.zeros((N, 4))
  ab = np.c_[a_flat, b_flat, zeros]
  # We launch the prediction on the grid 
  c = model.predict(ab)

  cc = c.reshape(aa.shape)
  plt.contourf(aa, bb, cc, cmap='bwr', alpha=0.5)

# Our model can only do 3 cuts, so the plot shows indeed
# that several blue points have to be allowed in the decision
# due to the limited capacity of the model
df.plot(kind='scatter', c='target', x='lat', y='lon', cmap='bwr')
plot_decision_boundary(model)

# %%
# We try to improve the decision tree model
# First of all we can make it deeper
model = DecisionTreeClassifier(max_depth=4, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm,
             index=["Miss", "Hit"],
             columns=['pred_Miss', 'pred_Hit'])
print(classification_report(y_test, y_pred))
df.plot(kind='scatter', c='target', x='lat', y='lon', cmap='bwr')
plot_decision_boundary(model)
# Indeed we get way better with max_depth=4, and it does not get any better
# %%
# I check for overfitting
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")
# So the generalization is very good
# %%
# Now we test 4 other models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
models = [
    RandomForestClassifier(n_estimators=100, random_state=0),
    #Logistic regression will fail because it can only draw straight lines/hyperplanes
    LogisticRegression(),
    SVC(),
    GaussianNB()
]

def train_eval(model):
  X_train, X_test, y_train, y_test = train_test_split(X, y,
      test_size = 0.3, random_state=0)
  start = time.time()
  model.fit(X_train, y_train)
  dt = time.time() - start
  y_pred = model.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  pd.DataFrame(cm,
               index=["Miss", "Hit"],
               columns=['pred_Miss', 'pred_Hit'])
  # print(classification_report(y_test, y_pred))
  fig, ax = plt.subplots(figsize=(8, 6))
  df.plot(kind='scatter', c='target', x='lat', y='lon', cmap='bwr', ax=ax)
  plot_decision_boundary(model)
  plt.title(f"Decision boundary for {model.__class__.__name__}, train acc: {model.score(X_train, y_train):.3f}, test acc: {model.score(X_test, y_test):.3f}, time to train = {dt:.3f} seconds")
  plt.text(1,-1,str(cm), fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

for model in models:
  train_eval(model)

# %%
# Churn is a classification task that means we want to predict if a customer will leave the service or not
df = pd.read_csv(url + "churn.csv")
df.head()

# %%
y = df["Churn"]
# The other columns are in O (others)
O = df.drop("Churn", axis=1)
# X is the rest of the features that are numerical (not categorical)
X = O.select_dtypes(include=[np.number])
X

# %%
#Let's do some pairplots to look at the data
sns.pairplot(pd.concat([X, y], axis=1), hue='Churn')




# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
models = [ 
  KNeighborsClassifier(),
  SVC(kernel="linear"),
  SVC(kernel="rbf"),
  GaussianProcessClassifier(1.0 * RBF(1.0)),
  DecisionTreeClassifier(),
  RandomForestClassifier(),
  MLPClassifier(max_iter=1000),
  AdaBoostClassifier(),
  GaussianNB(),
  QuadraticDiscriminantAnalysis(),
]

def new_train_eval(model):
  X_train, X_test, y_train, y_test = train_test_split(X, y,
      test_size = 0.3, random_state=0)
  start = time.time()
  model.fit(X_train, y_train)
  dt = time.time() - start
  y_pred = model.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  pd.DataFrame(cm,
                index=["Miss", "Hit"],
                columns=['pred_Miss', 'pred_Hit'])
  print(classification_report(y_test, y_pred))
  print(f"Time to train = {dt:.3f} seconds")
  print(cm)

for model in models:
  print('Model is', model.__class__.__name__)
  new_train_eval(model)

# %%
"""
### Exercise 2


- load the churn dataset `churn.csv`
- assign the `Churn` column to a variable called `y`
- assign the other columns to a variable called `features`
- select numerical columns with `features.select_dtypes` and asign them to a variable called `X`
- split data into train/test with test_size=0.3 and random_state=42
- modify the `train_eval` function defined earlier to test and compare different models and hyperparameters combinations.

You can find a list of models available [here](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html).

"""

# %%


# %%
"""
### Exercise 3

Define a new function that also keeps track of the time required to train the model. Your new function will look like:

```python
def train_eval_time(model):
  # YOUR CODE HERE
  
  return model, train_acc, test_acc, dt
```
"""

# %%
from time import time

# %%
