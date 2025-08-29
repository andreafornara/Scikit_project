# %%
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn as sklearn

from sklearn.metrics import r2_score

# Now we do some exercises on a big dataset kindly offered by sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
# %%
url = "https://raw.githubusercontent.com/zerotodeeplearning/ztdl-masterclasses/master/data/"
df = pd.read_csv(url + 'weight-height.csv')
df.head()
# %%
# Calculate size based on number of features
n_features = len(df.select_dtypes(include='number').columns)  # exclude categorical columns (for example "names", not present here)
height = 10/ n_features

sns.pairplot(
  df,
  height=height,
  kind='scatter',
  diag_kind='kde',
  plot_kws={'alpha': 0.7, 's': 10},      # for scatter: transparency and marker size
  diag_kws={'fill': True}               # for KDE: shaded curves
)
plt.suptitle("Pairplot of Weight and Height", y=1.02)

# %%
# Now we define our features (inputs, stuff that we know)
X = df[['Height']].values  # double brackets to get a 2D array (matrix) instead of a 1D array (vector)
# Now we define our targets (output, what we want to predict)
y = df['Weight'].values # single brackets to get a 1D array (vector)
#Caveat : it is 1D because in this case we only have one target variable, in general one
# can use y = df[['class_A', 'class_B', 'class_C']].values to get a 2D array (matrix) for multiple target variables
# %%
#We split the data into training and testing sets
#We want to train our model on the training set and evaluate it on the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, random_state=0)
#The model is defined
linear_model = sklearn.linear_model.LinearRegression()
linear_model.fit(X_train, y_train)
# %%
print('The training score is: ', linear_model.score(X_train, y_train))
print('The test score is: ', linear_model.score(X_test, y_test))
# %%
y_pred_test = linear_model.predict(X_test)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(X_test, y_test, alpha=0.7, s=10)
plt.plot(X_test, y_pred_test, color='red')
plt.title("Model coef: {:0.3f}, Intercept: {:0.2f}".format(linear_model.coef_[0], linear_model.intercept_))
plt.xlabel("Height")
plt.ylabel("Weight")
plt.grid()

# %%
plt.scatter(y_test, y_pred_test)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")

m = y_test.min()
M = y_test.max()

plt.plot((m, M), (m, M), color='red')

# %%
"""
### Exercise 1: multiple features

More features: `sqft`, `bdrms`, `age`, `price`

- load the dataset `housing-data.csv`
- visualize the data using `sns.pairplot`
- add more columns in the feature definition `X = ...`
- train and evaluate a Linear regression model to predict `price`
- compare predictions with actual values
- is your score good?
- change the `random_state` in the train/test split function. Does the score stay stable?
"""

# %%
url = "https://raw.githubusercontent.com/zerotodeeplearning/ztdl-masterclasses/master/data/"
df = pd.read_csv(url + 'housing-data.csv')
df.head()

# %% 
n_features = len(df.select_dtypes(include='number').columns)  # exclude categorical columns (for example "names", not present here)
height = 10/ n_features

sns.pairplot(
  df,
  height=height,
  kind='scatter',
  diag_kind='kde',
  plot_kws={'alpha': 0.7, 's': 10},      # for scatter: transparency and marker size
  diag_kws={'fill': True}               # for KDE: shaded curves
)
plt.suptitle("Pairplot of Housing Data", y=1.02)
# %%
def train_eval(random_state=0):
    X = df[['sqft', 'bdrms', 'age']].values  # double brackets to get a 2D array (matrix) instead of a 1D array (vector)
    y = df['price'].values  # single brackets to get a 1D array (vector)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=random_state)

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    # print('The training score is: ', model.score(X_train, y_train))
    # print('The test score is: ', model.score(X_test, y_test))

    # In this case we are probably overfitting the model!
    y_pred_test = model.predict(X_test)

    return r2_score(y_test, y_pred_test), r2_score(y_train, model.predict(X_train))

random_states = np.linspace(0,1000,1001, dtype = int)
train_scores = []
test_scores = []

for state in random_states:
    train_score, test_score = train_eval(random_state=state)
    train_scores.append(train_score)
    test_scores.append(test_score)

# %%
#histogram of the scores
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(train_scores, bins=30, alpha=0.7, label="Train Scores")
ax.hist(test_scores, bins=30, alpha=0.7, label="Test Scores")
ax.set_xlabel("R^2 Score")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid()

# Comment : The problem seems to be related to the number of data points, we need more data to consistently get good R^2 scores. How many data points do we have?
print('The scores are so bad because N_data =', df.shape[0])
# %%



#give me the dataset as a pandas df
california_housing = fetch_california_housing()
print(california_housing.DESCR)
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.Series(california_housing.target)
X = X[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']]
# %%
#Let's plot X
sns.pairplot(X, height=2.5, aspect=1.2, kind="scatter", diag_kind="kde", plot_kws={"alpha": 0.7, "s": 10}, diag_kws={"fill": True})

# %%
# We start again with a linear evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
model = sklearn.linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
print('The training score is: ', model.score(X_train, y_train))
print('The test score is: ', model.score(X_test, y_test))

# %%
#plot the actual vs predicted values
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7, s=10)
#Now I plot the "perfect line", such that, if I was perfect with my model, all points would lie on this line
plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color='red')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.grid()

# %%
#Now we can do something fancier, evaluate different models on the same dataset!
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7, s=10)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    m = y_test.min()
    M = y_test.max()
    plt.plot((m, M), (m, M), color='red')
    R_square = r2_score(y_test, y_pred_test)
    R_square_train = r2_score(y_train, model.predict(X_train))
    plt.title(f"Actual vs Predicted Values ({model.__class__.__name__}) - R^2: {R_square:.2f}, R^2 Train: {R_square_train:.2f}")
    plt.grid()
    plt.show()

#without tuning RandomForest I get overfitting. Notice that random forest is not performing well on extremal values.
for model in [LinearRegression(), Ridge(), Lasso(), RandomForestRegressor(n_estimators=100,
    max_depth=5,                 # Limit depth to prevent overly specific trees
    min_samples_leaf=10,         # Require more data to make a leaf
    max_features='sqrt',         # Reduce number of features considered per split
    random_state=42), GradientBoostingRegressor(n_estimators=100)]:
    evaluate_model(model, X_train, X_test, y_train, y_test)