# image-recognition-machine-learning-project

This is a 3-people-team-based project, I have contributed 40% of this project.

##Introduction: 
This project focuses on an image recognition problem based on the MNIST Fashion database, which collected a large number of images for different types of apparel. Each image is divided into small squares called **pixels** of equal area. Within each pixel, a brightness measurement was recorded in grayscale and the brightness values range from 0 (white) to 255 (black). We are given train data with 60,000 rows and 50 columns and test data with 10,000 rows and 50 columns. For this project, our group constructed 90 different machine learning models based on the training data to classify apparel in the test dataset.  Each model is applied in three sets of distinctive samples sizes of 500, 1000, 2000. 

The **goal** of the project is to **identify the best model that provides highest predictive power with fewest computation time and sample size**. At the end of the report, you will find a scoreboard that will evaluate each machine learning model based on the following formulas: 

**Points = 0.25 * A + 0.25 * B + 0.5 * C**

**A** is the the proportion of the training rows
**B** is the running time divided by a number of 60
**C** is the proportion of the predictions on the testing set that are incorrectly classified. 

The **best result** will have a minimal point which suggests best predictive accuracy with fewest computational power and sample size. Overall, our group explored 10 different machine learning techniques, which include: (1) Multinomial logistic regression, (2) K-Nearest Neighbours, (3) Classification Tree, (4) Random Forest, (5) Ridge Regression, (6) Lasso Regression, (7) Support Vector Machines, (8) Boosting Model, (9) Neural Networks, and (10) Ensemble Model. The Ensemble model is calculated based on averaging the time and accuracy of the 3 best models listed above.

##Machine Learning Techniques: 

### Model 1: 
**Multinomial logistic regression** (MLR) model is a classification method that generalizes logistic regression to multiclass problems. This model is used when the dependent variable is multinomial. The prediction formula of multinomial logistic regression model is $Pr[Y_i=1]=\frac{e^{\beta_1X_i}}{\Sigma_{k=1}^{K}e^{\beta_kX_i}}$. So this model uses a linear combination of the observed features and some problem-related parameters to estimate the probability of each particular value of the dependent variable. The multinom function in nnet package can build this model and there is no need to select any parameters manually. The advantage of this model is that it's easy to understand and the running time of this model is very small. But the accuracy of this model is not very high, because it's too simple to capture all features of training data.
 
### Model 2:  
Next, we explore **K-Nearest Neighbours**, which uses 'feature similarity' for prediction. The advantages of this model is its simplicity and adaptability to any distribution. Since the goal is to predict the category of clothes based on pixel image values (which ranges from 0 to 255), our group believe K-Nearest Neighbours serves as good technique since it is relatively easy to implement and K parameter can be tuned to give better prediction.

We use kknn function to perform K-Nearest Neighbor classification from kknn package. Initially, we run the function on default parameter to see how it performs. After, our group uses parameter tuning via leave-one-out method from train.kknn function to find the optimal k and kernel per the code below: 

xval = train.kknn(label~.,data= dat, nkmax = 10, kernel = c("optimal","rectangular","inv", "gaussian","triangular"), scale = TRUE)

The result shows that k = 5 and kernel = 'triangular' provides the best result and so we stick with these two parameters for our algorithm. We decide to stick to the default parameter for distance, which is calculated based on Minkowski distance. Although K-Nearest Neighbour is relatively simple to implement and is quite effective on predicting noisy and large dataset, it is known for its poor predictive performance. For this reason, we decided to perform run xval or parameter tuning on big dataset only once as we know that this will not provide the best result for our model. 

### Model 3:

**Decision tree** also known as CART (Classification and Regression Tree) is the third technique we used to create our predictive model. CART is a simple machine learning technique that is widely used for both metric (regression problem) and non-metric (classification problem) outcomes. The first split point is called *Root* node, the split after the first is called *Branches*, and the end node is called *Leaves*. Decision tree are well-established because the model is easy to build and provides high interpretability. The first split suggests the best predictor variable for the dependent outcome; hence, it is relatively easy to understand. Due to its interpretability, decision tree is widely used when the goal of the project revolves around interpretability over predictive performance.
 
As mentioned above, the disadvantage of decision tree is predictive performance. This is because decision tree generally partitions data into smaller and smaller subsets (branches) and hence at its end node, or leaves node, the data that is left contain only a few, similar data points. With this in mind, decision trees often overfits the data, meaning it performs exceptionally well on train data but poorly on unseen data. Being aware of this issue, our team decided to perform cross validation to overcome the overfitting problem. To put this in context, when cross validation is performed, the trees will continue to split until the error does not decrease significantly enough and then it will stop (Jake Hoare, 2018). 
 
Although we are well aware of decision tree relatively low predictive performance, we still decided to run the model for several reasons. First, since the goal of this project is not to produce a highest predictive model, but the one with high-enough accuracy with the least computational time and smaller sample size, we thought it will be a good idea to explore how decision tree model would perform. Second, since we are building a randomForest model, we believe decision tree is a good first step which will serve as a baseline for our more complex model like randomForest. 

In order to perform decision tree technique, we use rpart function from rpart package. Since the dependent variable, label, is a factor, type needs to be set to 'class.' In additional to type, complexity parameter (cp) need to be adjusted to improve predictive accuracy and reduce overfitting. Cp is used to prune the decision tree to make sure the model provides the optimal tree size without overfitting the training data. In other words, at the optimal cp, the decision tree will stop splitting. We find the optimal cp utilizing caret package per the code below:

trControl = trainControl(method="cv",number = 5)
tuneGrid = expand.grid(.cp = seq(0.001,0.1,0.001))
cvModel = train(label~.,data=train,method="rpart",
            	trControl = trControl,tuneGrid = tuneGrid)
cvModel$bestTune #0.001
 
The default cp of rpart is generally 0.01. We first build the model with default parameters and we use cross validation to find the optimal cp, which we found is 0.001. It helps improve our model so we decided to stick to 0.001. Again, the main purpose of building a decision tree is to find a baseline for random forest model. For this reason, our group decided to perform cross validation only once as we know that decision tree model will not provide the best performance for our prediction. 

### Model 4
Next, we move to **Random forest**, an ensemble machine learning classification method with high predictive performance. We like random forest because it construct a multitude of decision trees during training and output the class that is the mode of the classification. Also, random forest fixes the problem of overfitting that occurs in decision tree model. Since random forest uses bagging to build B trees from Bootstrap samples, the model ranks the importance of observed variables in a natural way and for this reason, random forest is able to generate a very good prediction from large train dataset. 

We use randomForest function in randomForest package. The default number of tree in this model is 500. Per the code below, we started building the model with default parameters and plot error of the model, which shows that the error becomes stable when number of trees is bigger than 300: 

mod.rf<-randomForest(the.formula,data=train,importance=TRUE, proximity=TRUE,ntree=300,mtry=12)
plot(mod.rf)

And then we build model with ntree=300, we choose the mtry between 1 and 49 that minimizes the prediction error of the model. Finally, we decide to use ntree=300, mtry=12 in our random forest model per the code below: 

n <- length(names(dat_2000_2))
err<-numeric(n-1)
for (i in 1:(n-1)){
model <- randomForest(the.formula, data = dat_2000_3, mtry = i)
err[i] <- mean(model$err.rate)
}
which(err==min(err))

The advantage of using randomForest is its high predictive power and easy to implement. Random forest is also good at dealing with data with many features. Also, the running time of this model is not very long as it builds trees parallely. The model performance is pretty good even if the observed variables are not linear. The disadvantage of this model is its low interpretability, high computational power, and the likelihood of still overfitting the dataset.

For Model 5 an 6, we decided to explore regularized regression techniques: Ridge and Lasso Regression. 

### Model 5

**Ridge regression** is a biased regression method for analyzing multiple regression data that suffers from multicollinearity. It can reduce the standard errors by adding a degree of bias to the regression estimates. Since we have 49 variables, problems we can meet are over-defined model, outliers and multicollinearity generated by our sampling method. Hence, ridge regression can be applied to reduce the effects of these problems. In ridge regression model, regularization avoids overfitting by shrinking the coefficients of correlated predictors towards each other. For these reasons, we believe that ridge regression will be a good fit for this dataset. Again, to serve as baseline to compare with the other more powerful machine learning techniques.   

To perform a ridge regression model, we use glmnet function with alpha = 0. Since the output is multiclass, we set family equal to 'multinomial.' 

A such model is ridge regression: 
$RSS+\lambda \Sigma_{i=1}^{p} \beta_i^{2}$

The tuning parameter in ridge regression is lambda. We use 10-fold  cross validation to choose the optimal  lambda per the code below: 

cv.ridge = cv.glmnet(x,y,alpha=0, family = "multinomial", type.multinomial = "grouped") opt_lambda <- cv.ridge$lambda.min

The lowest point in the curve indicates the optimal lambda. The best fit models with best lambda are also in the last row of the output of the ridge regression, which explains why we pick the last column in our code to get best prediction. 

Although ridge regression helps prevent overfitting problems the disadvantage is its computational cost and compromised accuracy. 

### Model 6

Another regularization technique is **Lasso regression**. Similar to ridge regression, lasso regression also helps avoid overfitting by shrinking the coefficient estimates and are generally use for the process of feature selection. The only difference between ridge and lasso regression is how the shrinkage method is calculated.  With a small modification of shrinkage penalty from ridge regression, lasso regression perform an absolute value to shrinkage penalty instead of squaring it, which allows the algorithm to force some of the coefficients equal to zero (Lala, 2019).  

The goal of ridge and lasso regression, like linear and logistic regression, is to minimize the sum of squares errors. However, both ridge and lasso have a shrinkage penalty, meaning that it penalizes over complicated models that provide no additional value to predictive performance.
 
Lasso regression is widely use as this machine learning technique fits well with linear, logistic, multinomial, poisson, and Cox regression models. (Hastie & Qian, 2014). The advantage of lasso regression is its ability to work with large number of predictors and give us an idea of which predictors to choose from. However, that comes with the cost of high computational power and poor predictive performance.
 
In order to perform lasso regression, our group use glmnet function with alpha = 1 from glmnet package. We use default function and set family = "multinomial" because the dataset is multiclass. Surprisingly though, even with a small sample size, lasso and ridge took relatively long to compute and was not able to provide as good predictive power as we anticipated. 

### Model 7

Next, we move to **Support Vector Machine** which constructs a hyperplane or set of hyperplanes in a high-dimensional space, which can be used for classification. A good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any points. The larger the margin, the lower the generalization error of the classifier. By defining a kernel function k(x,y) selected to suit a specific problem, the dot products of pair of input data vectors may be computed easily in terms of the variables in the original space. The hyperplanes are defined as the set of points whose dot product with a vector in that space is constant.

To create svm model, our group use svm function in e1071 package. The parameter 'type' in this function can be selected from 'C-classification', 'nu-classification', 'one-classification' according to the type of y. And we also could choose several kernel functions: linear, polynomial, radial and sigmoid. We build svm function with 3 kinds of type and 4 kinds of kernel function and find that when type is 'C-classification' and kernel function is 'radial', our model has highest accuracy. Then we use tune.svm function to choose best gamma and cost per the code below: 

svm_test <- function(x,y,test){
   type <- c('C-classification','nu-classification','one-classification')
   kernel <- c('linear','polynomial','radial','sigmoid')
   pred <- array(0, dim=c(nrow(test),3,4))
   errors <- matrix(0,3,4)
   dimnames(errors) <- list(type, kernel)
   for(i in 1:3){
 	for(j in 1:4){
   	pred <- predict(object = svm(x, y, type = type[i], kernel = kernel[j]), newdata = test[,2:50])
       errors[i,j]<-percentage.correctly.classified(pred,test$label)
   	}
 	}
   return(errors)
   }

tuned<-tune.svm(the.formula,data=dat_2000_1,type='C-classification', kernal ='radial',gamma = 10^(-6:-1),cost = 10^(1:2))
gamma=tuned$best.parameters$gamma
cost=tuned$best.parameters$cost

There are four main advantages of using svm model. First, it uses regularization parameter to avoid overfitting. Second, it uses the kernel trick to build model and provide good predictive performance. Third, it's an efficient method because it's a convex optimization problem. Finally, it can generate approximation bound of test error rate and the number of mistakes to build the model. The disadvantage of using svm, however, is computational cost. When the training data is big, this model can take a very long time to compute. This problem, however, didn't appear in our model, because our maximum sample size is only 2000.

### Model 8

Next, we explore **Generalized Boosted Regression Models**, which is a machine learning ensemble meta-algorithm for reducing bias and also variance (Wikipedia, 2019). It's another general approach for improving prediction performance that can be applied to many statistical methods. 

In our model, we use boosting to fit trees to the residuals from the previous tree. Here we have three tuning parameters: number of trees, shrinkage parameter (controls the speed of the process), and number of splits (interaction.depth). For shrinkage parameter, we use the the default value of 0.001. For the number of splits, typically it should be set to a small number. Considering the running time penalty and accuracy, we set it to 3. Generally, we use cross-validation to set n.tree. In this report, we use 5-fold cross validation to fit the best number of trees for prediction per the code below:

Gbm.cv = gbm(label ~ ., distribution = "multinomial", data = dat, n.trees = 500, interaction.depth = 3,shrinkage = 0.001,cv.folds = 5, n.cores = 1);
best_ntree = gbm.perf(Gbm.cv)

According to the cross-validation results and in consideration of the penalty of running time, we set n.tree as 500.Since our response is a factor, we set the distribution as multinomial. While the output of gbm model is a factor, we have to convert label in the test dataset to factor to evaluate the percentage of accuracy. 

The advantages of boosting are accuracy and effectiveness. The disadvantages of boosting model is that it requires extra work and extra data to perform cross-validation since it requires careful tuning of three parameters. Since we don't have large dataset and is constraint with the penalty of running time, we may not make the best of boosting models. 

### Model 9

The last model we decided to explore before ensembling is **Neural networks**. Neural networks are known to produce high predictive power but also comes at a cost of high computational power. This is because it involves machine learning or the process in which a company learns to perform a task based on training samples. Based on the definition provided from MIT website, "most of today's neural nets algorithms, data are organized in layers of nodes and each individual node is connected to several nodes in the layer beneath it, from which it receives data, and several nodes in the layer above it, to which it sends data" (Hardesty & MIT News Office, 2017).  
 
In MINIST fashion dataset, we feed in pixels that correlate to a particular apparel label, and neural nets find patterns that consistently correlate with that label and predict the label accordingly. We use nnet function from nnet package to create neural networks algorithm.
 
Since nnet function works best on normalized dataset, we divide our dataset by 255 to get the range to 0 and 1.In nnet function, softmax needs to be set to True for classification problem. We first run nnet with default parameters, and then we use cross validation to change parameters for *size* and *decay*. Size is the number of units in the hidden layer of neural network and decay is the parameter to help avoid overfitting. Per the code below, we use cross validation techniques to find the optimal size (10) and decay (0.1) for the model:
 
no_cores <- detectCores()
cl <- makeCluster(no_cores)
registerDoParallel(cl)
grid_nn <- expand.grid(size = seq(from = 1, to = 10, by = 1),
                    	decay = seq(from = 0.1, to = 0.5, by = 0.1))
fitControl <- trainControl(method = "repeatedcv",
                       	number = 10,
                       	repeats = 5,
                       	classProbs = TRUE)
nnetFit <- train(train2000_norm, make.names(train2000$label),
             	method = "nnet",
             	metric = "Accuracy",
             	trControl = fitControl,
             	tuneGrid = grid_nn)
 
With small sample size of just 2000, nnet was able to provide our group a really high predictive power and surprisingly low computational power. Again, the advantage of neural networks is high predictive power. With big dataset, that usually comes with a cost of high computation power and low interpretability. 

### Model 10


Last but not least, we decided to **Ensemble** the model by averaging time and accuracy of the three best models: (1) Support Vector Machine, (2) RandomForest, and (3) Neural Networks. The advantage of using this ensemble model method is that it is easy to perform and require low computational power. Instead of averaging accuracy, if we assign weight and utilize ensemble model to make new predictions, it might provide higher accuracy to our model. However, that also come with the cost of high computational power. 

## Scoreboard

## Discussion
Again, the scoreboard in the table above is calculated based on the following formulas: 

**Points = 0.25 * A + 0.25 * B + 0.5 * C**

**A** is the the proportion of the training rows
**B** is the running time divided by a number of 60
**C** is the proportion of the predictions on the testing set that are incorrectly classified 

As the scoreboard table suggests, **Support Vector Machine and Ensemble Model** serve as the best models for our group with a score of 0.0982. The sample size that provides the best accuracy is 2000 from train sample "dat_2000_3." 

Since the formula gives more weight to the predictive power of the model, it does not come as a surprise to why *Support Vector Machines* and *Ensemble Model* provide the best result. This is because *Support Vector Machine* is able to predict with high accuracy in a matter of few seconds, especially in such a small dataset in this project. As for *Ensemble Model*, the mean was calculated based on the three models with best time and predictive power. 

The points system identified above definitely impact our best result and the way our group runs our models. Since we are penalized on computational time, our group have to make sure that all the models provide highest accuracy without being overly complex (as it will take time to run). If, however, the goal of this report is to identify the model with best accuracy possible, the result in our scoreboard will be completely different. For instance, instead of trying to minimize the complexity of the model, our group will utilize different techniques to tune the parameters of different machine learning models, particularly, random forest, support vector machines, generalized boosted regression models, and neural networks to minimize errors. All these tuning models, again, come at the expense of more computational time and bigger sample size. 

All in all, if our group has computational resources to explore different sample size and variety of models, we will be able to better classify apparel in the test dataset than what is seen above. However, by doing so, our main concern will be shifted from minimizing sample sizes and computational power to finding ways to avoid overfitting the train dataset. Hence, our goal will be to offer the optimal complexity that is able to provide highest predictive accuracy of unseen data than what was done in this particular report. 

## References

Advantages and disadvantages of SVM. Retrieved from
https://stats.stackexchange.com/questions/24437/advantages-and-disadvantages-of-svm

Boosting (machine learning). (2019, February 19). Retrieved from https://en.wikipedia.org/wiki/Boosting_(machine_learning)

Hardesty, L., & MIT News Office. (2017, April 14). Explained: Neural networks. Retrieved March 13, 2019, from http://news.mit.edu/2017/explained-neural-networks-deep-learning-0414

Hastie, T., & Qian, J. (2014, June 26). Glmnet Vignette. Retrieved March 13, 2019, from https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html

Hoare, J. (2018, November 07). Machine Learning: Pruning Decision Trees. Retrieved March 13, 2019, from https://www.displayr.com/machine-learning-pruning-decision-trees/

Lala, V., PhD. (2019, March 13). Feature Selection: Applied Analytics Framework and Methods I. Lecture presented in NY, New York.

Minkowski distance. (2018, October 06). Retrieved from https://en.wikipedia.org/wiki/Minkowski_distance

Multinomial logistic regression. Retrieved from
https://en.wikipedia.org/wiki/Multinomial_logistic_regression

Random forest. Retrieved from
https://en.wikipedia.org/wiki/Random_forest

Shilane, D. (2019, March 11). Lecture 6 : A Grand Tour of Machine Learning. Lecture.
Support vector machine. Retrieved from
https://en.wikipedia.org/wiki/Support-vector_machine
