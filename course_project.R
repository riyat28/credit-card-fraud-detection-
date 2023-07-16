data <- read.csv("D:/SUBJECT/DS COURSE PROJECT/creditcardfraud/creditcard.csv")
View(data)
summary(data)
data$Class <- factor(data$Class , labels = c(0,1))
summary(data)

sum(is.na(data))

table(data$Class)
prop.table(table(data$Class))

labels = c("legit" , "fraud")
labels <- paste(labels,round(100*prop.table(table(data$Class)),2))
labels <- paste0(labels,"%")
labels

pie(table(data$Class),labels , col = c("cyan","red"),main = "Distribution of classes")

predict <- rep.int(0,nrow(data))
predict <- factor(predict, levels = c(0,1))

install.packages("caret")
library(caret)
confusionMatrix(data=predict,reference = data$Class)

library(dplyr)
set.seed(1)
data <- data %>% sample_frac(0.1)

table(data$Class)
prop.table(table(data$Class))

library(ggplot2)
ggplot(data = data, aes(x = V1,y = V2,col = Class))+
  geom_point() + 
  theme()+
  scale_color_manual(values=c('blue4','red4'))

install.packages('caTools')
library(caTools)

set.seed(123)

data_sample = sample.split(data$Class,SplitRatio = 0.80)
train_data <- subset(data,data_sample == TRUE)
test_data <- subset(data,data_sample == FALSE)

dim(train_data)
dim(test_data)

#ROS

table(train_data$Class)

prop.table(table(train_data$Class))

n_legit <- 22750
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit # = 22750/0.50

install.packages('ROSE')
library(ROSE)
oversample_data <- ovun.sample(Class ~.,
                               data = train_data,
                               method = "over",
                               N = new_n_total,
                               seed = 2019)

oversampled_data <- oversample_data$data
table(oversampled_data$Class)

pie(table(oversampled_data$Class),labels=c("legit" , "fraud") , col = c("cyan","red"),main = "Distribution of classes")

ggplot(data = oversampled_data, 
      aes( x = V1 , y = V2, col = Class))+
  geom_point(position = position_jitter(width=0.1))+
  theme()+
  scale_color_manual(values=c('blue4','red4'))

# RUS
table(train_data$Class)

n_fraud <- 35
new_frac_fraud <- 0.50
new_n_total<- n_fraud/new_frac_fraud

undersample_data <- ovun.sample(Class ~.,
                               data = train_data,
                               method = "under",
                               N = new_n_total,
                               seed = 2019)

undersampled_data <- undersample_data$data
table(undersampled_data$Class)

pie(table(undersampled_data$Class),labels=c("legit" , "fraud") , col = c("cyan","red"),main = "Distribution of classes")

ggplot(data = undersampled_data, 
       aes( x = V1 , y = V2, col = Class))+
  geom_point(position = position_jitter(width=0.1))+
  theme()+
  scale_color_manual(values=c('blue4','red4'))

#ROS and RUS
table(train_data$Class)

sampling <- ovun.sample(Class ~.,
                        data = train_data,
                        method = "both",
                        seed = 2019)

sampled_data<-sampling$data
table(sampled_data$Class)
ggplot(data = sampled_data, 
       aes( x = V1 , y = V2, col = Class))+
  geom_point(position = position_jitter(width=0.1))+
  theme()+
  scale_color_manual(values=c('blue4','red4'))

#SMOTE
install.packages("smotefamily")
library(smotefamily)

table(train_data$Class)

n0 <- 22750
n1 <- 35
r0 <- 0.6

ntimes <- ((1 - r0)/r0)*(n0/n1) - 1

smote_res <- SMOTE(X = train_data[,-c(1,31)],
                   target = train_data$Class,
                   K = 5,
                   dup_size = ntimes)

smoted <- smote_res$data

colnames(smoted)[30] <- "Class"

prop.table(table(smoted$Class))

ggplot(train_data, aes(x = V1 , y = V2, color = Class))+
  geom_point()+
  theme()+
  scale_color_manual(values = c('red','black'))

ggplot(smoted, aes(x = V1 , y = V2, color = Class))+
  geom_point()+
  theme()+
  scale_color_manual(values = c('red','black'))

#install.packages('rpart.plot')
#library(rpart)
#library(rpart.plot)

#cart_model <- rpart(Class~. , smoted)
#rpart.plot(cart_model, extra = 0,type = , tweak = 1.2)



#Cluster based sampling
#install.packages('survey')
#library(survey)

#table(train_data$Class)
#cluster <- sample(1:10,2)
#sampled <- subset(train_data$Class,cluster %in% cluster)
#table(sampled)

#RAW DATA LR
#logistic_model <- glm(Class~.,data = train_data, family = "binomial")
#test_data$predicted_class <- predict(logistic_model, newdata=test_data, type = "response") 
#test_data$predicted_class <- ifelse(test_data$predicted_class >= 0.5, 1 , 0)
#confusionMatrix(table(test_data$predicted_class, test_data$Class))

#Linear Model
logistic_model <- glm(Class~.,data = sampled_data, family = "binomial")
test_data$predicted <- predict(logistic_model, newdata = test_data, type = "response")
test_data$predicted <- ifelse(test_data$predicted >= 0.5, 1, 0)
confusionMatrix(table(test_data$predicted, test_data$Class))
library(pROC)
roc_obj <- roc(test_data$Class, test_data$predicted)
#par(mar = c(5, 4, 4, 2) + 0.1)
plot(roc_obj, type = "1", main = "ROC Curve for Credit Card Fraud Detection")


#DECISION Tree 
library(rpart)
library(rpart.plot)
tree_model <- rpart(Class~., data = sampled_data, method = 'class')
rpart.plot(tree_model, main = "Credit Card Fraud Detection")


#Random Forest
install.packages("randomForest")
library(randomForest)
library(caret)
rf_model <- randomForest(Class ~ ., data = sampled_data, ntree = 100)
predictions <- predict(rf_model, newdata = test_data)
confusionMatrix(predictions, test_data$Class)
plot(rf_model)


# Train the KNN model
library(caret)
library(class)
library(tidyverse)
model <- train(Class~.,data = smoted, method = "knn", trControl = trainControl(method = "cv", number = 5))
predicty <- predict(model, test_data)
confusionMatrix(predicty, test_data$Class)

ggplot(test_data, aes(V1, V2, color = Class, shape = predicty)) +
  geom_point() +
  ggtitle("KNN for Credit Card Fraud Detection") +
  xlab("V1") + ylab("V2")


creditcard_norm <- smoted %>%
  select(-Class) %>%
  scale() %>%
  as.data.frame() %>%
  cbind(Class = )

