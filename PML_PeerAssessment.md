Practical Machine Learning Course Project
========================================================

# Preprocessing and Cleaning Data
First, load in the data.  

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rpart)
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
data<-read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testData<-read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!",""))
```


Divide the training data into subsets, one to train the classifiers and one to use as a validation set.

```r
trainIndex<-createDataPartition(data$X, p=0.6, list=FALSE)
training<-data[trainIndex,8:160]
testing<-data[-trainIndex,8:160]
```


It is necessary to clean the training data.  First, remove the columns that have near zero variance.  

```r
NZV <- nearZeroVar(training, saveMetrics=TRUE)
training<-training[,NZV$nzv==FALSE]
```

The second data cleaning item performed is removing those columns that have over 30% of their values as NA.


```r
tmpTrain<-training
for(i in 1:length(training)){
    if( sum(is.na (training[,i]))/ nrow(training) >=.7){
        for(j in 1:length(tmpTrain)){
            if( length(grep(names(training[i]), names(tmpTrain)[j])) ==1){
                tmpTrain <-tmpTrain[,-j]
            }
        }
    }
}

training<-tmpTrain
```

Lastly, we guarantee that our validation set and the testData set contain the same columns as the training set.


```r
testing<-testing[colnames(training)]
testData<-testData[colnames(training[,-53])]
```


# Creating Classifiers

## Prediction Tree

The first classifier tried is a prediction tree.  The overall accuracy rate of 73.48% indicates that a more complex classifier is necessary, as the out-of-sample error rate is 26.52%.

```r
rp<- rpart(classe ~ ., data=training, method="class")
rp_pred<-predict(rp, testing, type="class")
confusionMatrix(rp_pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1992  257   23   69   13
##          B   76  910  177  118  147
##          C   47  208 1032  113  115
##          D   85   87   93  870  100
##          E   18   66   54  116 1062
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7475         
##                  95% CI : (0.7377, 0.757)
##     No Information Rate : 0.2826         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6801         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8981   0.5955   0.7484   0.6765   0.7390
## Specificity            0.9357   0.9180   0.9253   0.9444   0.9604
## Pos Pred Value         0.8462   0.6373   0.6812   0.7045   0.8070
## Neg Pred Value         0.9589   0.9037   0.9452   0.9371   0.9426
## Prevalence             0.2826   0.1947   0.1757   0.1639   0.1831
## Detection Rate         0.2538   0.1160   0.1315   0.1109   0.1353
## Detection Prevalence   0.2999   0.1820   0.1930   0.1574   0.1677
## Balanced Accuracy      0.9169   0.7568   0.8369   0.8104   0.8497
```

## Random Forest

The second classifier tried is a random forest.  The overall accuracy rate for this classifier is 99.25% with a confidence interval of (.9903, .9943).  This corresponds to a 0.75% out-of-sample error rate.


```r
set.seed(73729)
rf<-randomForest(classe ~ ., data=training, method="class")
rf_pred<-predict(rf, testing, type="class")
confusionMatrix(rf_pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2217    8    0    0    0
##          B    1 1513   16    0    0
##          C    0    7 1363   16    3
##          D    0    0    0 1270    5
##          E    0    0    0    0 1429
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9929          
##                  95% CI : (0.9907, 0.9946)
##     No Information Rate : 0.2826          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.991           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9995   0.9902   0.9884   0.9876   0.9944
## Specificity            0.9986   0.9973   0.9960   0.9992   1.0000
## Pos Pred Value         0.9964   0.9889   0.9813   0.9961   1.0000
## Neg Pred Value         0.9998   0.9976   0.9975   0.9976   0.9988
## Prevalence             0.2826   0.1947   0.1757   0.1639   0.1831
## Detection Rate         0.2825   0.1928   0.1737   0.1618   0.1821
## Detection Prevalence   0.2835   0.1950   0.1770   0.1625   0.1821
## Balanced Accuracy      0.9991   0.9937   0.9922   0.9934   0.9972
```


# Predicting Values for Test Data

Below are the predictions for the test data and the code used to create the submission files.


```r
testDataPred<-predict(rf, testData, type="class")
testDataPred
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```



```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(testDataPred)
```
