
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# not Splitting the dataset into the training and test sets
# not feature scaling

# fiting linear regression to the database
lin_reg = lm(formula = Salary ~., data = dataset)

# fitting polynomial regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level3 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., data = dataset)

# visualising the linear regression results
# install.packages('ggplots')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

# visualising the polynomial regression results
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# Predicting a new result with linear regression
y_pred_lin = predict(lin_reg, data.frame(Level = 6.5))

# Predicting a new result with polynomial regression
y_pred_poly = predict(poly_reg, data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4))



