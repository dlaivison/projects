# Projects : Enron analysis

#Udacity nanodegree  - Machine Learning - Project
# 1. Main Goals of this project

The target of this project was to create a machine learning algorithm with a good precision and accuracy in order to detect possible POI (person of interest) related to Enron case. The dataset was provided by Udacity and it contains about 146 registers. Each register has the following format :
```
>>> print data_dict["ELLIOTT STEVEN"]
{'salary': 170941, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 211725, 'exercised_stock_options': 4890344, 'bonus': 350000, 'restricted_stock': 1788391, 'shared_receipt_with_poi': 'NaN', 'restricted_stock_deferred': 'NaN', 'total_stock_value': 6678735, 'expenses': 78552, 'loan_advances': 'NaN', 'from_messages': 'NaN', 'other': 12961, 'from_this_person_to_poi': 'NaN', 'poi': False, 'director_fees': 'NaN', 'deferred_income': -400729, 'long_term_incentive': 'NaN', 'email_address': 'steven.elliott@enron.com', 'from_poi_to_this_person': 'NaN'}
```

The Enron case came to the media in 2001  and is one of most famous cases of fraud involving an energy company in America. In my analysis my focus was not to be distracted by certain patterns and focus mainly in the key term related and taken as the main result of fraud: money. I also considered the relations between Persons of Interest (POI) and other workers too. But considering that all account information have been already processed it could be the main indicator of other POI`s too.

In the beginning of my analysis there were 3 main outliers that were overfitting the data. The data from the main POI’s such as Kenneth Lay and Jeffrey Skiling were so far from the other non standard POI’s that they need to be removed. Another Outlier that could be considered highly as a POI was Mark Frevert. His net_worth was considered too high and in the same level as Lay and Skilling, that his status as POI should be considered as true. His data has been also removed from my dataset in order to create a perfect condition where other POI’s and non POI’s could be better detected.


As I stated before “cash” would be the key in my investigation. Furthemore we will going to see that I created a variable called net_worth (that contains all the informations related to payments) in order to check the possible connections with POI’s.

The total numbers of persons investigated in the dataset provided is 146. From data perspective there are in the dataset xx users identified as POI and others 

Sumary statistics of the dataset    :
```
[root@localhost final_project]# ./general_statistics.py
                      Dataset Size: 146
 Number of POI=true in the dataset: 18
Number of POI=false in the dataset: 128
                            Others: 0

I have created the script general_statistics in order to compute information about the dataset 

[root@localhost final_project]# ./general_statistics.py
                      Dataset Size: 146
 Number of POI=true in the dataset: 18
Number of POI=false in the dataset: 128
                            Others: 0
                            
 -------------------- Statistics about financial variables -----------------------------------------------------
Financial variable salary :                     % of info provided 65.07 % (Data informed: 95 Not Informed: 51  Total: 146 )
Financial variable deferral_payments :          % of info provided 26.71 % (Data informed: 39 Not Informed: 107  Total: 146 )
Financial variable total_payments :             % of info provided 85.62 % (Data informed: 125 Not Informed: 21  Total: 146 )
Financial variable loan_advances :              % of info provided 2.74 % (Data informed: 4 Not Informed: 142  Total: 146 )
Financial variable bonus :                      % of info provided 56.16 % (Data informed: 82 Not Informed: 64  Total: 146 )
Financial variable restricted_stock_deferred :  % of info provided 12.33 % (Data informed: 18 Not Informed: 128  Total: 146 )
Financial variable deferred_income :            % of info provided 33.56 % (Data informed: 49 Not Informed: 97  Total: 146 )
Financial variable total_stock_value :          % of info provided 86.3 % (Data informed: 126 Not Informed: 20  Total: 146 )
Financial variable expenses :                   % of info provided 65.07 % (Data informed: 95 Not Informed: 51  Total: 146 )
Financial variable exercised_stock_options :    % of info provided 69.86 % (Data informed: 102 Not Informed: 44  Total: 146 )
Financial variable other :                      % of info provided 63.7 % (Data informed: 93 Not Informed: 53  Total: 146 )
Financial variable long_term_incentive :        % of info provided 45.21 % (Data informed: 66 Not Informed: 80  Total: 146 )
Financial variable restricted_stock :           % of info provided 75.34 % (Data informed: 110 Not Informed: 36  Total: 146 )
Financial variable director_fees :              % of info provided 11.64 % (Data informed: 17 Not Informed: 129  Total: 146 )
```
From the information about some features with percentagem of information below 50% could be completed discharged but as I created the variable net_worth (including all the financial info), I removed them and net_worth was now being considered in the final analysis





# 2. About features used in this project

I used all financial features possible (about 14) in order to create a new variable called “net_worth”.  Just an exercise I also scaled the net_worth variable in order to have a better visualization in my plots. 

In order to engineer the variable net_worth, I have created a new numpy array and stored all the sum into it. After I need to concatenated the result into my_dataset variable.

##net_worth variable
This variable was created in order to sum all financial values and store in a numpy array

The idea behind to use net_worth is because some persons didn’t share enough information about their financial  data. But in term of fraud the high management has shared such info in the reports, so there would be at least some connection with “cash” that could be the key to find another POI. So I decided to consider any financial information in order to have a better understanding of the dynamics in such company.  I also did other exercises using the number of e-mails sent by Poi to such a person or vice-versa. But I thought that “money’ was a better indicator of financial fraud even because there are some communications that are made not using electronic mails (e-mails) and are better to do in loco.

## Variables to_poi_message_ratio and from_poi_message_ratio
The variables above were created in order to estimate the percentage of messages received by POI's or sent to POI's. The idea was to have a non financial variable that could indicate also some relevance to determine if a person is POI or non POI.

I used features_importance in order to verify the main labels that had real importance during the classification. As you can see the financial feature other has major weight on the classification.
```
Rank of features
0.167670 : other
0.117007 : total_payments
0.113114 : expenses
0.081226 : salary
0.079338 : exercised_stock_options
0.079130 : bonus
0.078201 : total_stock_value
0.077080 : restricted_stock
0.064582 : deferred_income
0.057949 : long_term_incentive
0.043895 : restricted_stock_deferred
0.040255 : deferral_payments
0.000551 : director_fees
0.000000 : loan_advances
```
But features_importance only works for tree type so I have changed the preliminary analysis in order to use SelectKbest.The results pointed to the same features, but it has automatized my proccess.

I have run with more than 5 features but the results for precision and recall were lower. So I have decided to keep only the 4 more relevant.
    The initial 4 more relevants features were choosen, using SelecKbest method and after the features_list was updated.

## After added the variables to_poi_message_ratio and from_poi_message_ratio
```
Rank of features
0.350057 : other
0.326995 : to_poi_message_ratio
0.167909 : total_stock_value
0.155040 : bonus
```
The routine to adjust a feature scalling was created but it was not used due the fact most of all values didn't change the values of precision and recall after adjusting the scale.  


# 3.  Algorithms used and reason for that


Thinking about the problem and how I would consider the best algorithm for this problem, I have no doubt that the algorithm should be a Tree kind. This is because the way I should consider someone POI or non POI would be considering several criterias. This is the way the Tree algorithm behaves  and it would not be a clustering k-means or even a linear regression … (Although linear could be applied, but there was no clear indication if a person was POI or non POI using such method).

I created an environment test and I have tested the performance of the following algorithms:
Initially GridSearchCV was used to optimize values but the results for prediciton and recall were below the expected
```
GridSearchCV(cv=None, error_score='raise',
       estimator=Pipeline(memory=None,
     steps=[('kbest', SelectKBest(k=4, score_func=<function f_classif at 0x7f098eb57848>)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=16,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'))]),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'kbest__k': [1, 2, 3, 4]}, pre_dispatch='2*n_jobs',
       refit=True, scoring=None, verbose=0)
        Accuracy: 0.79069       Precision: 0.27678      Recall: 0.22350 F1: 0.24730     F2: 0.23245
        Total predictions: 13000        True positives:  447    False positives: 1168   False negatives: 1553   True negatives: 9832
```
```        
GridSearchCV(cv=None, error_score='raise',
       estimator=Pipeline(memory=None,
     steps=[('kbest', SelectKBest(k=4, score_func=<function f_classif at 0x7f23dcc89848>)), ('tree', AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=32,
            max_features=None, max_leaf_nodes=None,
            m...None,
            splitter='best'),
          learning_rate=1, n_estimators=16, random_state=None))]),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'kbest__k': [1, 2, 3, 4]}, pre_dispatch='2*n_jobs',
       refit=True, scoring=None, verbose=0)
        Accuracy: 0.78469       Precision: 0.26006      Recall: 0.21650 F1: 0.23629     F2: 0.22400
        Total predictions: 13000        True positives:  433    False positives: 1232   False negatives: 1567   True negatives: 9768
```
So I decided to keep only the SelectKbest and use the results to directly attach to next Classifier.
The tunning parameters used after decided to use the DecisionTreeClassifier were the following:
max_depth = 32  -> for DecisionTreeClassifier
n_estimators=16 -> for AdaBoost Classifier


Initially the values were setup to default. During the tests I have increased the numbers of estimators and max_depth. As a result the precision and recall have improved.

Before  (defaut values)
```
AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1.0, n_estimators=50, random_state=None)
        Accuracy: 0.78962       Precision: 0.32272      Recall: 0.33450 F1: 0.32850     F2: 0.33208
        Total predictions: 13000        True positives:  669    False positives: 1404   False negatives: 1331   True negatives: 9596
```
After (Trying to optimize with higher values).
```
AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=64,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1, n_estimators=32, random_state=None)
        Accuracy: 0.79177       Precision: 0.32848      Recall: 0.33850 F1: 0.33342     F2: 0.33645
        Total predictions: 13000        True positives:  677    False positives: 1384   False negatives: 1323   True negatives: 9616
```

### Follows the results using other algorithms

### Using non SVM machines (Kmeans algorithm, clustering):
```
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=100,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=0, tol=0.0001, verbose=0)
        Accuracy: 0.80600       Precision: 0.21384      Recall: 0.17000 F1: 0.18942     F2: 0.17727
        Total predictions: 15000        True positives:  340    False positives: 1250   False negatives: 1660   True negatives: 11750
```



### Using Gaussian Naive bayes

``` 
GaussianNB(priors=None)
        Accuracy: 0.33673       Precision: 0.15693      Recall: 0.90900 F1: 0.26765     F2: 0.46413
        Total predictions: 15000        True positives: 1818    False positives: 9767   False negatives:  182   True negatives: 3233

```

### Using Decision tree
```
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
        Accuracy: 0.80620       Precision: 0.26588      Recall: 0.25750 F1: 0.26162     F2: 0.25913
        Total predictions: 15000        True positives:  515    False positives: 1422   False negatives: 1485   True negatives: 11578

```



### Using Random forest 
```
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=16, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
        Accuracy: 0.86167       Precision: 0.41379      Recall: 0.09000 F1: 0.14784     F2: 0.10670
        Total predictions: 15000        True positives:  180    False positives:  255   False negatives: 1820   True negatives: 12745
```


### Using Adaboost combined with Random Forest
```
AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=16, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False),
          learning_rate=1, n_estimators=10, random_state=None)
        Accuracy: 0.86567       Precision: 0.48538      Recall: 0.12450 F1: 0.19817     F2: 0.14625
        Total predictions: 15000        True positives:  249    False positives:  264   False negatives: 1751   True negatives: 12736
 ```       
   


### Optmized using Adaboost combined with DecisionTree
```
AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=32,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1, n_estimators=16, random_state=None)
        Accuracy: 0.81977       Precision: 0.40327      Recall: 0.35750 F1: 0.37901     F2: 0.36580
        Total predictions: 13000        True positives:  715    False positives: 1058   False negatives: 1285   True negatives: 9942

time processing test: 4.013 s
```




# 4. Tunning parameters



When I choose the DecisionTree  algorithm combined with Adaboost my idea was to allow the reclassification in a deeper level. I setup the Adaboost classifier to have 16 estimators (not more) in order to avoid delays during computational processing. For the DecisionTree I have set the variable max_depth=16 in order to have also a better precision.
The K values selected by SelectKbest were not the best based on f_classification parameter. So I decided to use the feature_importances as the main selector of the features in order to maximize the performance.

Adjusting the script and using the id's returned by feature_importances I could obtain the following result:
```
AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=32,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1, n_estimators=16, random_state=None)
        Accuracy: 0.81567       Precision: 0.44392      Recall: 0.41950 F1: 0.43136     F2: 0.42417
        Total predictions: 12000        True positives:  839    False positives: 1051   False negatives: 1161   True negatives: 8949
```

Tunning is an importante process in order to  have a better performance of the algorithm. But in order to have a good tunning you must know exactly the parameters you can choose to modify. In my case as I decided to use the DecisionTree , I knew that parameter max_depth would increase the ramifications of decision tree. So as I have several features this parameters could be used to reach a better precision. The feature_importances was computational calculated and it has been also used to improve my algorithm performance.

In terms of mathematical models, tunning is crutial when you have a statistical/probability algorithm created and you try to represent certain function based on the training system. Based on the approach you take you can be very closer to the original model and better predict its behaviour.


# 5. Validation


Validation is a way to estimate the machine learning algorithm trained performance. The method I used to validate my algorithm was based in criteria of precision , accuracy and recall. But mainly precision and accuracy. The provided tester.py script was used in order to evaluate some metrics regarding specifically the script.

In terms of performances indicators I choose 4:  Accuracy, Precision, Recall and time processing.  Precision is important and mainly for this Enron project because it shows the script has identified quite well the true positive cases. Although there are cases of false positives, but the script has performed about 44.39% of precision.  For Recall the performance was almost in the same level and the performace was about  41.95%.  And in terms of accuracy the results were 81.56% , which means also a good accuracy for a machine learning algorithm. The Time processing is also important because is also a indicator of performance in mathematical terms. So even a simpler algorithm can reach the same result as another complex algorithm, but a faster speed. This is important in case of process that demands short time response.

The script tester.py splits the data using the StratifiedShuffleSplit strategy.  This strategy seems very applicable for Enron dataset due the fact of missing information in some fields and the quantity of non POI. As this strategy splits randomly the data in the dataset, this could agregate more useful information than just splitting by percentage (such as used in crossvalidation). 


# 6. Evaluation metrics 

Using Adaboost classifier combined with DecisionTree I could obtain the following results from tester.py

```
AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=32,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1, n_estimators=16, random_state=None)
        Accuracy: 0.81567       Precision: 0.44392      Recall: 0.41950 F1: 0.43136     F2: 0.42417
        Total predictions: 12000        True positives:  839    False positives: 1051   False negatives: 1161   True negatives: 8949
```

According to the results my accuracy , considered the samples provided was about 81.97%  and the precision 40.32%. So the algorithm could predict the number of cases with a good accuracy, although the precision was medium. The fact of precision indicates about 50% means that there were some cases of false positives values. A similar value has been found for the recall 41.95% and indicates also cases of false negatives.









