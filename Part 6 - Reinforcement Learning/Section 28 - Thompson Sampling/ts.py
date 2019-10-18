
# Thompson Sampling (TS)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# TS algorithm takes 3 steps, 
# First step is at each round we need to consider two numbers for each ad i, the first number is the number of times the ad i got
# were one up to around n and the second number is the number of times the ad i got we were zero up to around n. We will consider 
# these parameters and declare the variables corresponding to these paramters and we can notice that if we compare the TS algorithm
# to the UCB algorithm, it is the same step one with different parameters because as we can notice here in the step one of the UCB
# we also consider 2 numbers and these two numbers are the number of times the ad i was selected up to round n and the sum of the 
# rewards of the ad i up to arounf n. 

# implementing TS
# Step 1:
import random
N = 10000           # number of samples
d = 10              # number of ads

# Step 1:
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d

# step 2 & 3:
# step 2 is for each ad i we take a random draw from this distribution below which is the beta distribution, it is because we have 2 important
# assumptions here which are related to Bayesian Inference:
# first assumption is this we suppose that each ad i gets rewards from Bernoulli distribution of parameter Theta i which is the probability of 
# success and you can picture this probablity of success by showing the ad to a huge amount of users and Theta i could be interpreted as the
# number of times the ad get rewarded one that is the number of success devided by the total number of times we selected the ad. so basically 
# Theta i is the probablity of success that is the probability of gettingreward one when we select the ad, and so the assumption is that each ad i
# gets reward zero and one from this Bernoulli distribution of parameter Theta i which is the probablity of success
# second assumpltion that the Theta i has a uniform distribution which is the prior distribution and then we use the Bayes rule to get to posterior
# distribution which is beta to our given the rewards that we got up to the round n, and so by using Bayes rule that is how we get this beta distribution
# here. and so by taking a random fraws of these beta distribution well seen these random draws represent nothing else and this probablity of success
# we get stradigy here which is to take the maximum of these random draws because the maximum of these random draws is approximating the highest probablity 
# of success, and this the whole idea behind the TS that we are trying to estimate these paramteres Theta that are the probablity of success of each of these 
# 10 ads then by taking these random draws and taking the highest of them we are estimating the highest probablity of success and this highest probablity of
# success corresponds to one specific ad at each round. So when we take these random dra that a specific round we might be wrong but when we take these
# random draws over thousands of rounds we will just based on the essence of probabilty we obtain over all the Theta that corresponds to the ad that has
# the highest probablity of success (reward 1). This is the step 3


ads_selected = []   # empty vector
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i]+ 1)            
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward


# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()