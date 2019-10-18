
# NLP

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
# to ignore the double quot we use the quoting = 3
# here we import the file which contains 1000 reviews ofsome resturants, and for each reviews we have this like
# column which tells us yes=1 or no=0
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# so our mission is to make machinery model that will predict for any new review if it
# is positive or negative. But mostly that is not the most important part here because 
# what we are doing for this dataset, we will be able to do it for other kinds of texts
# even when the purpose is very different.

# th first step of working with texts and making a model to predict things on text is to
# clean the text. Why we have to do this? it is because what will happen in the end is that
# we will create what's called a bag of words model or bag of words representation and this
# will consist of getting only the relevant words and the different reviews here. so that means
# that we will get rid of all the non useful words like 'the' or 'on' or others.. these are not
# relevant words because these are not the words that will help the ML  algorithm to predict if 
# the review is +ve or -ve.
# we will get rid of numbers, punctuations (ie dots) also we do stemming which consist of taking
# the route of some different versionsof a same word, why we do this , not to have too many words 
# at the end, also get rid of the capital and convert them to lower text.
# then we proceed to the last step of crating our bag of words, which it is the tokenization process.
# It splits all the different reviews into different words which using the text pre-processing. Then
# we will take all the words od the different reviews and we will attribute on column for each word
# we will have a lot of columns and then for each review each column will contain the number of times
# the associated word appears in the review. here we will get sparse matrix because we will get a lot 
# of zeros in the sparse matrix because for all the reviews most of the word will not be in the reviews
# so we will have to do something about it.

# clean the test
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# next step is stemming, consist only keeping the root of the different words to simplify the review
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # convert all upper case letters to lower case
    review = review.lower()
    # remove the non-significant words, the words that are not relevant into predicting whether the review
    # is -ve or -ve
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    # final step consists of joining back different words of this review list here composed of the elements
    # back to string
    review = ' '.join(review)
    corpus.append(review)

# creating the bag of words
# bag of words model: basically the first big step of NLP not only we cleaned all the reviews but
# we also created a curpus. so from this we are going to create a bag of words model. what we are going
# to do to create this bag of words model is just to take all the diferent words of these reviews 
# without taking twice or three times the duplicates or triplicates we are just taking all the different
# but unique words of these reviews and basically to create on clumn for each word. we will have a lot 
# of columns. so we will have the 1000 review rows and the columns will be correspond to each of
# the different words we find here in all the reviews in corpus. So each cell of this table will 
# correspond to one specific review and one specific word of this corpus. and in this cell we are 
# going to have a number and this number is going to be the number of times the word corresponding 
# to the column appears in the review.
# with this table with rows and columns, which is most columns will be zero this table called sparcse Matrix.
# and the fact that we have a lot od zeros is called sparsity and that is a very common notion in ML.

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
# get our independent variables
y = dataset.iloc[:, 1].values

# now we can build our model out of X, y
# each line in X corresponds to one specific review and for each of those reviews the spawns to one word
# and we get a zero if the word doesn't appear in the review and a one if the word appears in the review.
# And so basically that gives us a classification model because now we will traina ML model and so basically 
# that gives us a classification model because now we will train a machine or any model that will try to
# understand the correlations between the presence of the words and the reviews and the outcome is zero if it is 
# a -ve reviewand one if it is +ve review
# the common model we use for NLP are: Naive Bayes, DT classification, Random Forest Classification.
# we ar using Naive base here:
# Splitting the dataset into the training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# fitting classifie to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# predicting the Test set results
y_pred = classifier.predict(X_test)

# making the confusion Matrix (Compute confusion matrix to evaluate the accuracy of a classification)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# the result from cm:
# 55 correct predicitions of -ve reviews, 91 correct predicitions of +ve reviews
# 42 incorrect predicitions of +ve reviews, 12 incorrect predicitions of -ve reviews.
# so that means that out of 200 reviews our machine model made 55 + 91 = 146 correction predicitions
# and 42 + 12 = 54 incorrect predicitions
# and accuracy is (55+91) / 200 = 0.73 



























