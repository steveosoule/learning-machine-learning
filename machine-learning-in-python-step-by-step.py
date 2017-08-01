# http://machinelearningmastery.com/machine-learning-in-python-step-by-step/


# 1.2 Start Python and Check Versions
# ------------------------------------------------

# Python version
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

# Print Versions
# print('Python: {}'.format(sys.version))
# print('scipy: {}'.format(scipy.__version__))
# print('numpy: {}'.format(numpy.__version__))
# print('matplotlib: {}'.format(matplotlib.__version__))
# print('pandas: {}'.format(pandas.__version__))
# print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 2.2 Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# 3. Summarize the dataset

# 3.1 Dimensions of Dataset
# print(dataset.shape)

# 3.2 Peek at the Data
# print(dataset.head(20))

# 3.3 Statistical Summary
# print(dataset.describe())

# 3.4 Class Distribution
# print(dataset.groupby('class').size())

# 4. Data Visualization

# 4.1 Univariate Plots
# box plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()


# histograms
# dataset.hist()
# plt.show()

# 4.2 Multivariate Plots

# scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

# 5. Evaluate Some Algorithms

# 5.1 Create a Validation Dataset

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# 5.2 Test Harness

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# 5.3 Build Models

"""
Let’s evaluate 6 different algorithms:

	1. Logistic Regression (LR)
	2. Linear Discriminant Analysis (LDA)
	3. K-Nearest Neighbors (KNN).
	4. Classification and Regression Trees (CART).
	5. Gaussian Naive Bayes (NB).
	6. Support Vector Machines (SVM).
"""

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	# print(msg)

# Compare Algorithms
"""
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
"""

# 6. Make Predictions

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))