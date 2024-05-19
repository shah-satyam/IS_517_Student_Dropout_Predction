# Predicting Dropout Risk: A Comparative Analysis of Machine Learning Algorithms for Student Dropout
_Aryan Patil, Samarth Jain, Satyam Shah_

## 1. Background/History
### 1.1. Literature Review of dataset and analysis method
Student dropout is a universal challenge faced by educational institutions. It is beneficial to 
identify students who may be at risk of dropping out so that appropriate attention can be 
provided to address their academic issues before they reach a critical stage. We have 
obtained the student dropout data from UC Irvine's data repository. The dataset is rich in 
features, containing over 4,000 rows and 37 features. This provides us with an opportunity to 
explore the wider aspects of student's lives and their impact on academics.
In their research paper titled "Early Prediction of Student's Performance in Higher Education: 
A Case Study," Martins et al. (2021) address the issue of imbalanced classes by conducting 
a comparative study of different machine learning models. They use the same dataset and 
test models such as Logistic Regression, SVM, Decision Tree, and Random Forest. To tackle 
class imbalance, they apply methods such as SMOTE and ADASYN to their Logistic 
Regression model. In addition, they also test some boosting models such as Gradient 
Boosting, Extreme Gradient Boosting Logit Boost, and CatBoost. The results of their study 
suggest that boosting models perform better than standard machine learning models, but 
even they fail to correctly classify the minority classes.

### 1.2. Limitations of previous study or analysis method
Although the aforementioned study explores a range of machine-learning techniques to 
address class imbalance, it omits any discussion of feature selection and the variables used. 
Given that the dataset comprises 37 distinct features, selecting the appropriate ones is 
crucial to optimize the training of the machine learning models. Additionally, the study 
overlooks the potential benefits of utilizing neural networks. Our project aims to enhance 
their findings by refining feature selection techniques and experimenting with alternative 
machine-learning methods and strategies to address class imbalance.

## 2. Overview
This project aims to compare various data processing techniques, balancing methods, and feature
selection approaches. By systematically evaluating different combinations, we seek to identify
the optimal approach for a student dropout dataset.

### 2.1. Objective
The objective of this project is to enhance existing research on early prediction of student 
dropout in higher education by addressing the limitations of previous studies. Specifically, 
the project aims to refine feature selection techniques and experiment with alternative 
machine-learning methods and strategies to improve the accuracy of dropout prediction 
models. By leveraging a rich dataset containing over 4,000 rows and 37 features, the project 
aims to develop a robust predictive model capable of identifying at-risk students at an early 
stage, thereby enabling educational institutions to provide timely intervention and support 
to prevent student dropout.

### 2.2. Dataset
The dataset used for this project, sourced from UC Irvine's data repository, focuses on 
predicting student outcomes in higher education, where the target variable has three 
classes: dropout, enrolled, and graduated. With over 4,000 rows and 37 features, it
encompasses a wide range of information including demographic details, academic 
background, socio-economic factors, and behavioral indicators. Despite its richness, the 
dataset exhibits class imbalance, with varying proportions of students in each outcome
class.

### 2.3 Key Components
1. _Data Processing Techniques:_
   - Explore methods for data cleaning, transformation, and normalization.
   - Address class imbalance using balancing techniques.
2. _Feature Selection Methods:_
   - Investigate filter, wrapper, and embedded methods.
   - Select relevant features to improve model performance.
3. _Machine Learning Algorithms:_
   - Evaluate logistic regression (including Lasso and Ridge), random forest, and XGBoost.
   - Understand their strengths and weaknesses.
4. _Results and Conclusion:_
   - Compare performance metrics for each combination.
   - Determine the best approach based on empirical evidence.

_Note: Detailed project explanation is present in the [Project Report](IS_517_Project%20Report.pdf)_.
_The final report is based on the results form the following execution fo the analysis notebook: [Execution Results](Final%20Notebook%20Execution.html)_