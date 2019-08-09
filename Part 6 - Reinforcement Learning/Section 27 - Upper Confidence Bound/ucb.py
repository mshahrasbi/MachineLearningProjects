
# ucb 

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# for some car company marketing campaigns remember we had this business client of the social 
# network that puts ads on the social network and then we made these classification models to
# target the userson the social network most likely to buy this SUV that the car company launched
# at a very low price and basically to prepare this marketing campaign this car comapny prepared
# an ad that they would put on the social network, and what happened is that the department of 
# marketing prepared some different versions of the same ad, like puting the car in different scebarios 
# like for example one ad had car on a beautiful road and on the another version of the ad the car is on a 
# mountain and maybe on another version it's on a beautiful bridge. But the problem is that they prepared
# 10 great versions of the same ad the 10 versions of this ad look great. So they are actually not sure
# which ad to put on the social network, they want to put ad that will get the maximum clicks.So they need
# to put the ad that will lead to the best conversion rates. So what this car company did is that they hired
# us as a data scientist and they said ok I have 10 versionsof the ad. we have a limited budget to place
# the ads on the social network beacuse putting these ads on the socialnetwork costs them money. So they want us to 
# find the best strategy to quickly find out which version of this ad is the best for the user, that is
# which version of the ad will lead us to the highest conversion rate. 
# So we strat with no data, I know we have some data set here, but this is just data set for simulation
# beacuse what happens in real life, we are going to start experimenting with this ads by placing them
# on social network, the different versions of the ad and according to the results we observe we will change
# our stradtegy to place these ads on the social network.
# So we have 10 version of the same ad, and each time a user of the social network will log into his
# account we will place one version of these 10 ads and that will be around each time a user connects to
# its account we will show him one version of the ad and we will observe his response, if the the user
# clicks on the ad we will get a reward equal 1 otherwise 0.
# we are not going to show the different versions of the ads we choose or at random, there is going to be
# a specific startegy to do this. and the key thing to understand about reinforcment learning is that this
# startegy will depend at each round on the previous results we observed at the previous rounds. That is
# why reinforcement learning is also called online learning or interactive learning, beacuse the startegy is 
# dynamic it depends on the observations from the beginning of the experiment upto the present time.
# so we are going to build two algorithms the ucb and the Tompson sampling algorithms, and these algorithm
# dicide from here which version of the ad to show to the user.

# implementing UCB
# Step 1:
import math
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d

# step 2 & 3:
N = 10000           # number of samples
d = 10              # number of ads
ads_selected = []   # empty vector
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward


# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()































