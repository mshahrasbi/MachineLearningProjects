
# Apriori
# Data Preprocessing 
dataset = read.csv("Market_Basket_OPtimisation.csv", header = FALSE)

# this is not the dataset we are going to use, we are going to use to train our
# Apriori model. The reason is that the package we are going to use to build our
# Apriori model which is by the way the ARules package doesn't take a dataset like 
# this as input. What it takes as input is called a sparse matrix. What it is? It is
# matrix that contains a lot of zeroes. In machine learning you will encounter a lot of
# times the word sparsity that corresponds to a large number of zeroes. 
# so what we are going to do now is transform this dataset here into a sparse matrix.
# will what we are going to do is take all the different products in this dataset and 
# actually there are 120 products and we are going to attribute one column to each of 
# these 120 products.And the lines are still going to be the different transactions 
# corresponding to each of the 7500 customers that bought a basket of products during
# the whole week.

# create Sparse Matrix
# install.packages('arules')
library(arules)

dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)

summary(dataset)

itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
# what supports do we want to have all our different items in the roules so
# that the rules are relevant because for example if we go back to the plot
# we can see that we have a lot of products that are not pruchased very frequently
# and these specific products other products with small supports beacuse a few 
# transactions contain these products here so when you divide the number of 
# transactions containing this products by the total number of transactions then
# you will get a small support and you know since these products are not pruchased 
# very often ther are not very relevent for optimization problem because you know
# we want to optimize the sales. That's what we want to optimize rule is the revenue.
# and since the revenue is a linear combination of the different numbers of products
# where the coefficients are actually the prices of these products. well in order to 
# optimize the revenue we would need to optimize the sales of these products here that 
# are pruchased very often rather than these products here there are less pruchased
# And so what we need to choose here is the support that will only include the products
# on the left of this vertical bar here that will correspond to the minimum support.
# So how to choose the support, well we need to look at the products that are pruchased 
# rather frequently like at least 3 or 4 times a day (again that depends on your business
# goal). But what's for sure is that if we manage to find some strong rules about items
# that are bought at least 3 or 4 times a day then by associating them and placing them 
# togather customers will be more likely to put them in their basket and therefore more 
# of these products will be purchased and therfore the sales will increase. So that will be the
# starting point of how we are going to set the minimum support we are going to consider 
# the product there are purchased at least 3 or 4 times a day and then we will look at the rules 
# and of course if we are not convinced by the rules we will change this value of the support. UNtil we 
# satisfied with the rules and until we think it makes sense, and you know we can also try these
# rules within a certain period of time and then if we look at the impact on the revenue and if we 
# don't observe a meaningful increase in the sales revenue we can later change the support and the
# confidence to change the rules and then experience again until we find the strongest rules that 
# are in sales
# 3 (day) * 7 (week) / 7500 = 0.0028 (~ 0.003)
# 4 (day) * 7 (week) / 7500 = 0.0037 (~ 0.004)

# rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.8)) # no rules
# rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.4))
# rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.2))
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# visualising results
inspect(sort(rules, by = 'lift')[1:10])







