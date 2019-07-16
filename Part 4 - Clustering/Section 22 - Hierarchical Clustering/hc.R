# HC

# Import the Mall_Customer dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# using Dendrogram to find optimal number of clusters
dendrogram = hclust(dist(X, method = "euclidean"), method = "ward.D")
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fitting HC to the dataset
hc = hclust(dist(X, method = "euclidean"), method = "ward.D")
y_hc = cutree(hc, 5)

# visualising 
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of clients'),
         xlab = 'Annul Income',
         ylab = 'Spending Score')