# Install necessary packages if not already installed
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("gplots")) install.packages("gplots")
if (!require("dplyr")) install.packages("dplyr")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("reshape2")) install.packages("reshape2")
if (!require("rpart")) install.packages("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("caret")) install.packages("caret")
if (!require("randomForest")) install.packages("randomForest")
if (!require("e1071")) install.packages("e1071")

# Load the required libraries
library(ggplot2)
library(gplots)
library(dplyr)
library(tidyverse)
library(reshape2)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(e1071)

# Load the dataset
car_data <- read.csv("C:/Users/91709/OneDrive/Desktop/car_evaluation.csv")
print(dim(car_data))
print(summary(car_data))

# Examine the dataset
print(head(car_data, 10))
print(str(car_data))

# Rename columns
colnames(car_data) <- c("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
print(colSums(is.na(car_data)))

# Bar charts
ggplot(car_data, aes(x = class, fill = lug_boot)) + geom_histogram(stat = "count") + 
  labs(title = "Class Vs Luggage boot", y = "Frequency of Luggage boot", x = "Class")

ggplot(car_data, aes(class, fill = safety)) +
  geom_bar(position = position_dodge()) +
  ggtitle("Car class vs Safety") +
  xlab("Class") + 
  ylab("Safety")

ggplot(car_data, aes(class, fill = buying)) +
  geom_bar(position = position_dodge()) +
  ggtitle("Car class vs Buying Price") +
  xlab("Class") + 
  ylab("Buying Price")

# Density Plots
ggplot(car_data, aes(fill = as.factor(doors), x = persons)) + geom_density(alpha = 0.3)
ggplot(car_data, aes(fill = as.factor(maint), x = class)) + geom_density(alpha = 0.3) + facet_wrap(~class)

# Model 1: Decision Tree
set.seed(100)
train_test_split <- createDataPartition(y = car_data$class, p = 0.7, list = FALSE)
train_data <- car_data[train_test_split, ]
test_data <- car_data[-train_test_split, ]

# Train a decision tree model
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
decision_tree <- train(class ~ ., data = train_data, method = "rpart", 
                       parms = list(split = "information"), trControl = train_control, tuneLength = 10)

# Plotting decision tree
prp(decision_tree$finalModel, type = 3, main = "Probabilities per class")

# Predictions and accuracy
train_pred <- predict(decision_tree, train_data)
test_pred <- predict(decision_tree, test_data)

# Confusion matrix and accuracy
print(confusionMatrix(test_pred, as.factor(test_data$class)))

# Model 2: Random Forest
# Train a random forest model
random_forest <- randomForest(as.factor(class) ~ ., data = train_data, importance = TRUE)
plot(random_forest)
varImpPlot(random_forest, main = 'Feature Importance')

# Fine-tuning the model
random_forest_1 <- randomForest(as.factor(class) ~ ., data = train_data, ntree = 500, mtry = 3, importance = TRUE)

# Predictions and accuracy
train_pred1 <- predict(random_forest_1, train_data, type = "class")
test_pred1 <- predict(random_forest_1, test_data, type = "class")

# Confusion matrix and accuracy
print(confusionMatrix(test_pred1, as.factor(test_data$class)))

# Model 3: Naive Bayes
set.seed(123)
train_index <- sample(nrow(car_data), 0.7 * nrow(car_data))
train_data <- car_data[train_index, ]
test_data <- car_data[-train_index, ]

# Train a Naive Bayes model
nb_model <- naiveBayes(class ~ ., data = train_data)

# Predictions and accuracy
nb_pred <- predict(nb_model, newdata = test_data)
accuracy <- sum(nb_pred == test_data$class) / nrow(test_data)
print(paste("Naive Bayes accuracy:", round(accuracy, 4)))

