# Regression Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# not Splitting the dataset into the training and test sets
# not feature scaling

# fitting the Regression to the dataset
# Create your regressor here

# Predicting a new result
y_pred_poly = predict(regressor, data.frame(Level = 6.5))

# visualising the Regression Model results
# install.packages('ggplots')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# visualising the Regression Model results (for higher resolution and smoother curve)
x_grid = seq(min(dataset$Level), max(datasel$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')




