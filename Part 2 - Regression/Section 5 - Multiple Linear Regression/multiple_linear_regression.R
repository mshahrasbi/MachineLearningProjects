
# Import the dataset
dataset = read.csv('50_Startups.csv')

# enoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# splitting the dataset into the training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# fitting multiple linear regression to the training set
# regresor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
# regressor = lm(formula = Profit ~ .,
#               data = training_set)
regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)

# predicting the test set results
y_pred = predict(regressor, newdata = test_set)



