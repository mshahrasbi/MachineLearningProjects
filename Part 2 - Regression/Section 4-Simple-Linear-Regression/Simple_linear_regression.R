
# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# splitting the dataset into the training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling

# fitting simple linear regression to the training set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)
y_pred = predict(regressor, newdata = test_set)

# visualising the training set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs Experience (Traing set') +
  xlab('Years of Experience') +
  ylab('Salary')

# visualising the test set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs Experience (Test set') +
  xlab('Years of Experience') +
  ylab('Salary')
