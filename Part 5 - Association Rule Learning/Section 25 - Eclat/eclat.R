
# Eclat

# Data Preprocessing 
dataset = read.csv("Market_Basket_OPtimisation.csv", header = FALSE)


# create Sparse Matrix
# install.packages('arules')
library(arules)

dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)

summary(dataset)

itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

# visualising results
inspect(sort(rules, by = 'support')[1:10])