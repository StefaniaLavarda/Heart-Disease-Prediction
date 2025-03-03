# Heart Disease unsupervised and supervised project

## 1. Load required libraries
library(readr)
library(lattice)
library(ggplot2)
library(caret)
library(ROCR)
library(nnet)
library(randomForest)
library(e1071)
library(MASS)
library(mlbench)
library(plyr)
library(dplyr)
library(car)
library(PerformanceAnalytics)
library(corrplot)
library(ggpubr)
library(gplots)
library(tidyverse )
library(haven)
library(FactoMineR)
library(factoextra)

## 2. Load and Preprocess the Dataset

# Load the dataset
heart_disease_dataset <- read_csv("/Users/stefanialavarda/Desktop/statistical_project_1/heart.csv")
data <- heart_disease_dataset
                       
# Convert categorical variables to factors
categorical_vars <- c("Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", "HeartDisease")
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# Scaled numerical variables
numerical_vars <- c("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak")
data[numerical_vars] <- lapply(data[numerical_vars], scale)

# Perform FAMD 
res.famd <- FAMD(data, ncp = 5, graph = FALSE)

ind <- get_famd_ind(res.famd)
ind
# Visualize the individuals
fviz_famd_ind(res.famd, col.ind.sup= "lightblue", repel = TRUE)

# Visualize the variables
fviz_famd_var(res.famd , repel = TRUE )
# Contribution to the first dimension
fviz_contrib(res.famd, "var", axes = 1)
# Contribution to the second dimension
fviz_contrib(res.famd, "var", axes = 2)

fviz_screeplot(res.famd)
## 3. Exploratory Data Analysis (EDA)

# Summary statistics
View(data)
summary(data)

# Visualization of categorical variables

# Bar plot for Sex
ggplot(data, aes(x = Sex)) + 
  geom_bar(fill = "lightblue") + 
  theme_minimal() +
  labs(title = "Distribution of class", x = "Sex", y = "Count") 

# Bar plot for Chest Pain Type
ggplot(data, aes(x = ChestPainType)) + 
  geom_bar(fill = "lightblue") + 
  theme_minimal() +
  labs(title = "Distribution of class", x = "Chest Pain Type", y = "Count") 

# Bar plot for Resting ECG
ggplot(data, aes(x = RestingECG)) +
  geom_bar(fill = "lightblue") +
  theme_minimal() + 
  labs(title = "Distribution of class", x = "Resting ECG", y = "Count")

# Bar plot for Exercise Angina
ggplot(data, aes(x = ExerciseAngina)) +
  geom_bar(fill = "lightblue") +
  theme_minimal() + 
  labs(title = "Distribution of class", x = "Exercise Angina", y = "Count")

# Bar plot for ST Slope
ggplot(data, aes(x = STSlope)) +
  geom_bar(fill = "lightblue") +
  theme_minimal() + 
  labs(title = "Distribution of class", x = "ST Slope", y = "Count")

# Bar plot for Fasting BS
ggplot(data, aes(x = FastingBS)) +
  geom_bar(fill = "lightblue") +
  theme_minimal() + 
  labs(title = "Distribution of class", x = "Fasting BS", y = "Count")

# Count of Heart Disease cases
table(data$HeartDisease)

# Bar plot for Heart disease
ggplot(data, aes(x = HeartDisease)) + 
  geom_bar(fill = "lightblue") + 
  theme_minimal() +
  labs(title = "Distribution of class", x = "HeartDisease", y = "Count") +
  scale_x_discrete(labels = c("0" = "No Heart Disease", "1" = "Heart Disease")) 

# Bar plot for Heart Disease by Sex
ggplot(data, aes(x = Sex, fill = HeartDisease)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Proportion of Heart Disease by Sex", x = "Sex", y = "Proportion") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "lightgreen")) 

# Bar plot for Heart Disease by Chest Pain Type
ggplot(data, aes(x = ChestPainType, fill = HeartDisease)) +
  geom_bar(position = "fill") +
  labs(title = "Proportion of Heart Disease by Chest Pain Type", x = "Chest Pain Type", y = "Proportion") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "lightgreen")) +
  theme_minimal()

# Bar plot for Heart Disease by Resting ECG
ggplot(data, aes(x = RestingECG, fill = HeartDisease)) +
  geom_bar(position = "fill") +
  labs(title = "Proportion of Heart Disease by Resting ECG", x = "Resting ECG", y = "Proportion") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "lightgreen")) +
  theme_minimal()

# Bar plot for Heart Disease by ExerciseAngina
ggplot(data, aes(x = ExerciseAngina, fill = HeartDisease)) +
  geom_bar(position = "fill") +
  labs(title = "Proportion of Heart Disease by Exercise Angina", x = "ExerciseAngina", y = "Proportion") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "lightgreen")) +
  theme_minimal()

# Bar plot for Heart Disease by STSlope
ggplot(data, aes(x = STSlope, fill = HeartDisease)) +
  geom_bar(position = "fill") +
  labs(title = "Proportion of Heart Disease by ST Slope", x = "ST Slope", y = "Proportion") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "lightgreen")) +
  theme_minimal()

# Visualization of numerical variables

# Heart Disease distribution by Cholesterol level
data <- data %>% filter(Cholesterol > 0)

data_chol <- data %>% 
  mutate(Cholesterol_cohort = case_when(
    Cholesterol < 200 ~ 'Good Cholesterol Level', 
    Cholesterol >= 200 & Cholesterol < 240 ~ 'Borderline High', 
    Cholesterol >= 240 ~ 'High'
  ))

ggplot(data_chol, aes(x = Cholesterol_cohort, fill = HeartDisease)) +
  geom_bar(position = "dodge") +
  labs(x = "Cholesterol Level", y = "Count of Heart Disease", fill = "Heart Disease") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "lightpink"))

# Heart Disease distribution by Age
data_age <- data %>% 
  mutate(age_cohort = case_when(
    Age < 40 ~ 'Young', 
    Age >= 40 & Age < 60 ~ 'Adult', 
    Age >= 60 ~ 'Old'
  ))

ggplot(data_age, aes(x = age_cohort, fill = HeartDisease)) +
  geom_bar(position = "dodge") +
  labs(x = "Age", y = "Count of Heart Disease", fill = "Heart Disease") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "lightpink"))

# Heart Disease distribution by MaxHR 
ggplot(data, aes(x = MaxHR, color = HeartDisease)) +
  geom_density(size = 1) +
  scale_color_manual(values = c("0" = "lightblue", "1" = "lightpink"), 
                     name = "Heart Disease", 
                     labels = c("No", "Yes")) +
  labs(title = "Density Plot of MaxHR by Heart Disease", 
       x = "MaxHR", 
       y = "Density") +
  theme_minimal()

# Heart Disease distribution by Old peak
ggplot(data, aes(x = Oldpeak, color = HeartDisease)) +
  geom_density(size = 1) +
  scale_color_manual(values = c("0" = "lightblue", "1" = "lightpink"), 
                     name = "Heart Disease", 
                     labels = c("No", "Yes")) +
  labs(title = "Density plot of OldPeak by Heart Disease", x = "Old peak", y = "Density") +
  theme_minimal()

# Skeweness calculation 
skeweness_values <- data.frame(
  Feature = c('Age', 'Cholesterol', 'RestingBP', 'MaxHR','Oldpeak'),
  Skeweness = c(
    skewness(data$Age),
    skewness(data$Cholesterol),
    skewness(data$RestingBP),
    skewness(data$MaxHR),
    skewness(data$Oldpeak) )
)

print(skeweness_values)

# Skewness of Old peak
# Check for negative values 
data %>% filter(Oldpeak < 0)  %>% select(Oldpeak)
data <- data  %>% mutate(Oldpeak = abs(Oldpeak)) 

ggqqplot(data$Oldpeak)
skewness(data$Oldpeak)

ggdensity(data, x = "Oldpeak", fill = "lightgray", title = "Old peak") +
  stat_overlay_normal_density(color = "red", linetype = "dashed")

# Creating a new column for the log-transformation of Oldpeak
data <- data %>%
  mutate(log_Oldpeak = log(Oldpeak + 1)) 

# qq plot and density graph of the log transformation
ggqqplot(data$log_Oldpeak)

ggdensity(data, x = "log_Oldpeak", fill = "lightgray", title = "Logarithm of Old peak") +
  stat_overlay_normal_density(color = "red", linetype = "dashed")

skewness(data$log_Oldpeak)

# Skewness of Cholesterol
ggqqplot(data$Cholesterol)
skewness(data$Cholesterol)

ggdensity(data, x = "Cholesterol", fill = "lightgray", title = "Cholesterol") + 
  stat_overlay_normal_density(color = "red", linetype = "dashed")

data <- data %>% 
  mutate(log_Cholesterol = log(Cholesterol))

ggqqplot(data$log_Cholesterol)
skewness(data$log_Cholesterol)

ggdensity(data, x = "log_Cholesterol", fill = "lightgray", title = "Cholesterol") + 
  stat_overlay_normal_density(color = "red", linetype = "dashed")

## 4. Correlation Analysis

# Compute correlation matrix for numerical variables
my_data <- data %>% select(Age,RestingBP,Cholesterol,MaxHR,Oldpeak)
res <-cor(my_data) 
round(res, 2)

symnum(res, abbr.colnames = FALSE)

library(Hmisc)
rcorr(as.matrix(my_data))

# Scatter plot matrix
scatterplotMatrix(my_data, regLine = TRUE)

# Heatmap
col<- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = res, col = col, symm = TRUE, Colv = NA, Rowv = NA)

corrplot(res, type = "upper", 
         tl.col = "black", tl.srt = 45)

chart.Correlation(my_data, histogram=TRUE, pch=19)

# Does resting blood pressure differ significantly between those who 
# have heart disease and those who do not?
data$HeartDisease <- factor(data$HeartDisease,
                            levels = c(1,0),
                            labels = c("yes", "no"))
boxplot(RestingBP ~ HeartDisease, id=TRUE, data=data, main = "Resting Blood Pressure by Heart Disease Status")

plotmeans(RestingBP ~ HeartDisease, data=data)

ddply(data,~HeartDisease,summarise,mean=mean(RestingBP),sd=sd(RestingBP),n=length(RestingBP))

summary(aov(RestingBP ~ HeartDisease, data=data))

# Does cholesterol differ significantly between those who 
# have heart disease and those who do not?
data$HeartDisease <- factor(data$HeartDisease,
                                 levels = c(1,0),
                                 labels = c("yes", "no"))

boxplot(Cholesterol ~ HeartDisease, id = TRUE, data = data, main = "Cholesterol by Heart Disease Status")

plotmeans(Cholesterol ~ HeartDisease, data=data)

ddply(data,~HeartDisease,summarise,mean=mean(Cholesterol),sd=sd(Cholesterol),n=length(Cholesterol))

summary(aov(Cholesterol ~ HeartDisease, data=data))

# Does mean differ by class? T-test
t.test(Age~HeartDisease, alternative='two.sided', conf.level=.95, var.equal=FALSE, data=data)
t.test(RestingBP~HeartDisease, alternative='two.sided', conf.level=.95, var.equal=FALSE, data=data)
t.test(Cholesterol~HeartDisease, alternative='two.sided', conf.level=.95, var.equal=FALSE, data=data)
t.test(MaxHR~HeartDisease, alternative='two.sided', conf.level=.95, var.equal=FALSE, data=data)
t.test(Oldpeak~HeartDisease, alternative='two.sided', conf.level=.95, var.equal=FALSE, data=data)

# Chi-square test association between HeartDisease and categorical variables
mytab <- xtabs(~HeartDisease + Sex, data = data)
mytab
plot(mytab, col=c("lightpink", "lightblue"))
plot(t(mytab), col=c("lightpink", "lightblue"))
print (prop.table (mytab, 1)) 
print (prop.table(mytab, 2) )
Test <- chisq.test(mytab, correct=FALSE)
Test

mytab <- xtabs(~HeartDisease + ChestPainType, data = data)
mytab
plot(mytab, col=c("lightpink", "lightblue"))
plot(t(mytab), col=c("lightpink", "lightblue"))
print (prop.table (mytab, 1)) 
print (prop.table(mytab, 2) )
Test <- chisq.test(mytab, correct=FALSE)
Test

mytab <- xtabs(~HeartDisease + FastingBS, data = data)
mytab
plot(mytab, col=c("lightpink", "lightblue"))
plot(t(mytab), col=c("lightpink", "lightblue"))
print (prop.table (mytab, 1)) 
print (prop.table(mytab, 2) )
Test <- chisq.test(mytab, correct=FALSE)
Test

mytab <- xtabs(~HeartDisease + RestingECG, data = data)
mytab
plot(mytab, col=c("lightpink", "lightblue"))
plot(t(mytab), col=c("lightpink", "lightblue"))
print (prop.table (mytab, 1)) 
print (prop.table(mytab, 2) )
Test <- chisq.test(mytab, correct=FALSE)
Test

mytab <- xtabs(~HeartDisease + ExerciseAngina, data = data)
mytab
plot(mytab, col=c("lightpink", "lightblue"))
plot(t(mytab), col=c("lightpink", "lightblue"))
print (prop.table (mytab, 1)) 
print (prop.table(mytab, 2) )
Test <- chisq.test(mytab, correct=FALSE)
Test

mytab <- xtabs(~HeartDisease + STSlope, data = data)
mytab
plot(mytab, col=c("lightpink", "lightblue"))
plot(t(mytab), col=c("lightpink", "lightblue"))
print (prop.table (mytab, 1)) 
print (prop.table(mytab, 2) )
Test <- chisq.test(mytab, correct=FALSE)
Test

## 5. Split Dataset into Training and Testing Sets
set.seed(123)  
train_index <- createDataPartition(data$HeartDisease, p = 0.6, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# Train models
model_rf <- train(HeartDisease ~ ., data = train_data, method = "rf")
pred_rf  <- predict(model_rf, test_data)

model_svm <- train(HeartDisease ~ ., data = train_data, method = "svmLinear")
pred_svm  <- predict(model_svm, test_data)

model_nn <- train(HeartDisease ~ ., data = train_data, method = "nnet", trace = FALSE, linout = FALSE)
pred_nn  <- predict(model_nn, test_data)

model_logistic <- train(HeartDisease ~ ., data = train_data, method = "glm", family = "binomial")
pred_logistic  <- predict(model_logistic, test_data)

# Confusion matrices
cat("\nConfusion Matrix - SVM\n")
print(confusionMatrix(test_data$HeartDisease, pred_svm))

cat("\nConfusion Matrix - Random Forest\n")
print(confusionMatrix(test_data$HeartDisease, pred_rf))

cat("\nConfusion Matrix - Neural Network\n")
print(confusionMatrix(test_data$HeartDisease, pred_nn))

cat("\nConfusion Matrix - Logistic Regression\n")
print(confusionMatrix(test_data$HeartDisease, pred_logistic))

# ROC curves
pred_svm_rocr <- prediction(as.numeric(pred_svm), as.numeric(test_data$HeartDisease))
pred_rf_rocr  <- prediction(as.numeric(pred_rf), as.numeric(test_data$HeartDisease))
pred_nn_rocr  <- prediction(as.numeric(pred_nn), as.numeric(test_data$HeartDisease))
pred_logistic_rocr <- prediction(as.numeric(pred_logistic), as.numeric(test_data$HeartDisease))

roc_svm_perf <- performance(pred_svm_rocr, measure = "tpr", x.measure = "fpr")
roc_rf_perf  <- performance(pred_rf_rocr, measure = "tpr", x.measure = "fpr")
roc_nn_perf  <- performance(pred_nn_rocr, measure = "tpr", x.measure = "fpr")
roc_logistic_perf <- performance(pred_logistic_rocr, measure = "tpr", x.measure = "fpr")

roc_svm_df <- data.frame(FPR = unlist(roc_svm_perf@x.values),
                         TPR = unlist(roc_svm_perf@y.values),
                         Model = "SVM")

roc_rf_df  <- data.frame(FPR = unlist(roc_rf_perf@x.values),
                         TPR = unlist(roc_rf_perf@y.values),
                         Model = "Random Forest")

roc_nn_df  <- data.frame(FPR = unlist(roc_nn_perf@x.values),
                         TPR = unlist(roc_nn_perf@y.values),
                         Model = "Neural Network")

roc_logistic_df <- data.frame(FPR = unlist(roc_logistic_perf@x.values),
                              TPR = unlist(roc_logistic_perf@y.values),
                              Model = "Logistic Regression")

roc_data <- rbind(roc_svm_df, roc_rf_df, roc_nn_df, roc_logistic_df)

# Plot ROC curves
ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1) +
  geom_abline(linetype = "dashed", color = "red") +
  theme_bw() +
  labs(title = "ROC Curve Comparison - Heart Disease Data", 
       x = "False Positive Rate", 
       y = "True Positive Rate", 
       color = "Models") +
  theme(plot.title = element_text(hjust = 0.5))


## 6. Logistic Regression Model
logistic_model_age <- glm(HeartDisease ~ Age, data = data, family = binomial)
summary(logistic_model_age)
logLik(logistic_model_age)
(vcov = vcov(logistic_model_age)) # variance-covariance matrix

# Logistic regression visualization
data$fitted_values <- logistic_model_age$fitted

new_data <- data.frame(
  Age = data$Age,
  HeartDisease = as.numeric(data$HeartDisease)-1,
  fitted_values = logistic_model_age$fitted
)

ggplot(data, aes(x = Age, y = as.numeric(HeartDisease) - 1)) +
  geom_point(color = "blue") +
  geom_line(aes(y = fitted_values), color = "red", size = 1.5) + 
  labs(x = "Age", y = "Heart Disease Probability") +
  theme_bw()

# use PCA to detect the most importante variable feature and then used them for neural network
# if i don't use PCA, i have 11 predictors and only 510 features, troppo poche

# Random forest
library(tree)
library(ISLR)

tree.carseats=tree(HeartDisease~.,data=data)
summary(tree.carseats)

plot(tree.carseats)
text(tree.carseats,pretty=0)

tree.carseats

set.seed(101)
train=sample(1:nrow(data),400)
tree.carseats=tree(HeartDisease~.,data,subset=train)
plot(tree.carseats);text(tree.carseats,pretty=0)

tree.pred=predict(tree.carseats,data[-train,],type="class")
with(data[-train,],table(tree.pred,HeartDisease))

(184+121)/346

set.seed(101)
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass)
cv.carseats

plot(cv.carseats)

prune.carseats=prune.misclass(tree.carseats,best=9)
plot(prune.carseats);text(prune.carseats,pretty=0)

tree.pred=predict(prune.carseats,data[-train,],type="class")
with(data[-train,],table(tree.pred,HeartDisease))

(165+121)/346


library("rpart")
library("rpart.plot")
tree3<-rpart(HeartDisease~.,data[,-1])
printcp(tree3)
rpart.plot(tree3)
# Selecting the optimal subtree according to the 1-SE rule
tree4<-prune(tree3,cp=0.0001)
rpart.plot(tree4)


