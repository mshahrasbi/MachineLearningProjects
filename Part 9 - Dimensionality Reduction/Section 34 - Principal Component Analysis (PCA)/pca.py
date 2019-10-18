# PCA

# As you know in dimensionality reduction there are two techniques:
#    feature selection 
#    feature extraction
# we did feature selection in part 2 (Regression) when we implementd the backward elimination model to select the most relevant 
# features of our matrix of features, that is the features that explain the most the dependent variable and now we are starting 
# this new technique of dimensionality reduction which is feature extraction and PCA principal component analysis is one feature
# extraction technique. So as reminder let's say your matrix of features has m indepenedent variables, wel  what PCA will do is 
# that it will extract a smaller number of your independent variables but there are going to be new independent varaibles like
# new dimensions and these new independent variables extracted are going to be some new independent variables that explain the 
# most the variance of your dataset, and that is regardless of your depenedent variable, and that makes PCA an unsupervised mode
# in the sense that we don't consider the dependent varaible in the model, so that is PCA.

# remember in part 2 and 3 we worked with one or two independent variables, well that was for two specific purposes. The first purpose is
# that we needed a graphic visualization of our results. Ans since each independent variable corresponded to one dimension in the plot
# well we could visualize our results with at most two independent variables and the 2nd reason is that thanks to this PCA dimensionality
# reduction technique, well even if we have a lot of independent variables at the beginning well we can end up with much less independent
# variables but that is going to be relevant independent variables because these independent variables will explain the most the variance 
# of your dataset, and therefore since we can reduce this number of independent variables well we can end up with two or three independent
# variables and therefore visualize the results as we did in part 3.


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wine.csv')

# the original data set this dependent variable is not called customer_segment, this is actually the origin of the wine, but lets imagine that we
# as data scientist are working for one business owner and this one business owner gathered all these informations in this dataset and so first what
# this business owner did is that it gathered all the information of these independent variables here that are chemical's informations of several
# wines and this business owner applied some clut=stering technique to find some segments of customers that like a specific wine depending on the
# information of the wine and by applying these clustering techniques, this business owner identified 3 seqments of customers, so based on these information
# and its clustering techniques, well this one business oner managed to find some segments of customers each segment having a specific prefernce for 
# a specific wine, so basically this business oner found 3 of wines each type of one corresponding to one segment of customers and therefore 3 segments
# of customers, and why does it create added value for its business, well that's because now what this business owner can do is take all these informations
# of the wines as well as the information about the customer segments and make a classification model like logistic regression in which the independent 
# variables are all these variables and the different variable is the customer segments and therefore for each new wine it can predict to which customer
# segment it should recommend this wine. So the logistic regression model that we are going to build is going to return the customer segment that each new wine 
# should be recommended to. So that adds a lot of value for this business owner. But then if this business owner wants to have a clear visual look at the
# the prediction regions and the prediction boundary of the classification model that we are going to build to be able to see if the predictions are in the 
# right spot of the customer segments, well it can not be done with all these independent variables because of course we cannot represent these many independent 
# variables in one plot, so what we need to do is apply some dimensionality reduction techniques to extract two independent variables that explain the most the variance
# and then we will be able to see the prediction regions and the prediction boundary and therefore will clearly be able to see where the customer segments are 
# and where are these predictions that the customer segments are, according to the extracted features of all the informations of our independent variab;es, and
# remember these exta=racted features are called the PCA.
# So lets build our logistic regression classification model and let's apply PCA to reduce the dimensionality of our problem and evetually to visualize the results



X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
# This is the number of extracted features you want to get, that will explain the most the variants and depending on what variants you would
# like to be explained you will choose the right number of principal components, this vaiable is equal ad that is where we specify this number.
# but the problem now is that we know we want to get two principal components eventually to be able to visualize this training set result nd the
# test results but we don't know how much variance these two components explain, so we need to check that, you know we need to make sure that
# the two first principal components that explain the most variants don't explain it to low variance and therefore we are not giving input 2
# here, we are going to input none, because then we will create a vector that we are going to call explains variance and we are going to see the 
# cumulative variance explained by all the principal components,
# number of principal component = None
# pca = PCA(n_components = None) 
pca = PCA(n_components = 2) 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# so here we are going to create this explains variance vector that is going to contain the percentage of variance explained by each of the principal
# components that we extracted here, so explain variance and then the trick is to use a natural beauty of the PCA object so we take our object PCA and 
# that and then attribute that we want to take is the Explained variance ratio here that will give us a list of all principal components and we will
# get the percentage of variance explianed by each of them. by look at the explained_variance vector we can see the original 13 independent variables
# will it extracted 13 principal components, so these 13 components are independent variables but these are not the original independent varaibles that
# we had in our data set. these are the new extracted independent variable but that explained the most the variance, as you can see they are ranked from
# the first principal component that explians the most the variancedown to the 12th and last principal component that explians the least the variance.
# so that means that if we include one principal component that will explain 37% of the variance then if take 2 principal components that will explian 37 + 19 
# = 56 %of the variance and etc .. 
# so we want to take the first 2 principal components beacuse we want to get 2 dimensions in the visi=ualization of the training such result and therefore 
# we need 2 independent variables that is the 2 PC
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()