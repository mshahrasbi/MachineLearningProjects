# Artificial Neural Network

# Theano - open source numerical computation library, very efficient for fast numerical computation. This library
# can be run on CPU  but also on GPU. In terms of power and computation efficiency GPU is much more powerfull, it
# has many more cores and it is able to run a lot more floating points calculations per second.

# Keras - it is used to build deep neural networks in a very few lines of code. it is based on Theano and Tensorflow.

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# This is a classification problem, so we have some independent variables which are some informations about customers
# in bank and we are trying to predict a binary outcome for the dependent variable which is either one if  the customer
# leaves the bank and zero if the customer stays in the bank.

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# we first import keras lib, which we use it to build our deep NN based on tensorflow, we will import
# 2 modules as well; Sequential module that is required to initialize our NN and Dense module that is
# required to build the layers of our ANN
# Now we are ready to initialize he Neural Network (ANN). There are actually 2 ways of initializing a 
# deep learning module:
#   defining the sequence of layers
#   defining a graph
# so here we initializing our models by defining it as a sequence of layers. we just need to create an
# object of sequential class, this object is the model itself, that is the neural network that will have 
# a role of a classifier here because our problem is a classification problem where we have to predict a
# class.
# so now we are ready to add the different layers in this and the first is the hidden layer
# the 1st step is to randomly initialize the weights of each of the nodes to small numbers close to zero,
# that will be done through the Dense function, our first observation rules goes into the neural network and
# as we can see it this step two each feature is going to one input node, so we already know the number of 
# nodes we will have an input layer and this number is nothing else than the number of independent variables we 
# have in our matrix of features. so that means that in our input layer we will have 11 input nodes.
# the third step is for propagation, so from left to right the neurons are activated by the activation
# function in such a way that the higher the value of the activation function is for the neuron the more 
# impact this neuron is going to have in the network. That means the more it will pass on the signal from
# the nodes on the left to the nodes on the right. and so speaking of activation function which will have
# to do in this tutorial to define the first hidden layer is to choose an activation function and as a 
# reminder there are several activation functions, and the bets one based on experiments and based o research
# is the rectifir function, we also have sigmoid function which is good for output layer since using the 
# sigmoid function for output layer will allow us to get the probabilities of the different segments taht
# will get th probability that the classical one for each of the observation and even the new observations of
# the test set when we will make our predictions, so that means that for each observations of the test set
# we will get the probability that the customer leaves the bank and the probability that the customer stays 
# in the bank. So we are trying to build a segmentation model and by getting the probability thanks to the 
# sigmoid activation function fully output layer well we will be able to see which customers have the highest 
# probabilities to leave the bank. so we will even be able to make a ranking of the customers ranked by their
# probability to leave the bank.So you can segment your customers according to their probability to leave the
# bank and according to what you decide to do in terms of business constraints and business goals.
# we choose the rectifier activation function for hidden layers
# and choose the sigmoid activation function for output layer.
# in step 4 the algorithm compares the predictive result to the actual result and that generates an error and 
# then in step 5 this error is back  propagated in NN right to left and  all the weights are updated according
# to how much they are responsible for this generated error.
# step 6 where we repeate step 1 2 5 either after each observation or either after a batch of observations like 
# for example we do weights every 10 observations going into the network and finally step 7 when the whole training
# set pass through the ANN that makes an epoche and re repeat many more epoche. 
# 

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# units = output node (11 + 1)/2
# kernel_initializer = randomly init them with a uniform function, initialize the weights according to 
#       uniform distribution
# activation = rectifier function 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
# there is no input parameters for the snd layer since it knows what to expect because the first hidden
# layer was created, out put and initialize weight and activation function paramteres will be the same as first layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# the out will be only one node, activation function is going to be sigmoid function since we are in output layer
# we are making deo demographic segmetation model we want to have probabilities for outcome, because we want to know
# the probability of each customer leaves the bank.
# by the way if we are dealing with dependent variable that has more than 2 categories then you will need to change two things
# in this layer, - output to number of categories (classes) - then activation function need to be changed to Soft
# Max, it is kind of sigmoid functionbut applied to a dependent variablethat has more than two categories.
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
# compiling the whole ANN that is basicly applying SGD on the whole ANN

# optimizer = is simply the algorithm we will use to find the optimal set of weightsin the NN
#   because you know we defined our NN with the different layers but the weights are still only initialized.
#   so we have to apply some sort of algorithm to find the best weights; 

# loss = lost function within the SGD (adam) algorithm, because if you go deeper into the mathematical details
#   of SGD you will see that it is based on a loss function that you need to optimize to find the optimal weights.
#   if your dependent variable has a binary outcome then using binary_crossentropy function, if dependent variable 
#   has more than two outcome then categorical_crossentropy.

# metrics = it is just a criterion that we choose to evaluate your model, when the weights are updated after each 
#   observation or after each batchof many observations the alogorithm uses this accuracy criterion to improve the models
#   performance
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# batch_size = number of observations after which you want to update the weights
# epoch = is basically around when the whole training ser passed through the ANN. and in reality training ANN
#   consists of applying step 1 to 6 over many epochs
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)



# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# From CM we can see that 1542 + 139 correct, and 266 + 53 are incorrect 
# so the accuracy is the number of correct predications divided by the total number of predictions
# (1542 + 139) / 2000 = 0.8405




