# Projects : Enron analysis

#Udacity nanodegree  - Machine Learning - Project
# 1. Main Goals of this project

The target of this project was to create a machine learning algorithm with a good precision and accuracy in order to detect possible POI (person of interest) related to Enron case.  The Enron case came to the media in 2001  and is one of most famous cases of fraud involving an energy company in America. In my analysis my focus was not to be distracted by certain patterns and focus mainly in the key term related and taken as the main result of fraud: money. I also considered the relations between Persons of Interest (POI) and other workers too. But considering that all account information have been already processed it could be the main indicator of other POI`s too.

In the beginning of my analysis there were 3 main outliers that were overfitting the data. The data from the main POI’s such as Kenneth Lay and Jeffrey Skiling were so far from the other non standard POI’s that they need to be removed. Another Outlier that could be considered highly as a POI was Mark Frevert. His net_worth was considered too high and in the same level as Lay and Skilling, that his status as POI should be considered as true. His data has been also removed from my dataset in order to create a perfect condition where other POI’s and non POI’s could be better detected.


As I stated before “cash” would be the key in my investigation. Furthemore we will going to see that I created a variable called net_worth (that contains all the informations related to payments) in order to check the possible connections with POI’s.


# 2. About features used in this project

I used all financial features possible (about 14) in order to create a new variable called “net_worth”.  Just an exercise I also scaled the net_worth variable in order to have a better visualization in my plots. 

In order to engineer the variable net_worth, I have created a new numpy array and stored all the sum into it. After I need to concatenated the result into my_dataset variable.

The idea behind to use net_worth is because some persons didn’t share enough information about their financial  data. But in term of fraud the high management has shared such info in the reports, so there would be at least some connection with “cash” that could be the key to find another POI. So I decided to consider any financial information in order to have a better understanding of the dynamics in such company.  I also did other exercises using the number of e-mails sent by Poi to such a person or vice-versa. But I thought that “money’ was a better indicator of financial fraud even because there are some communications that are made not using electronic mails (e-mails) and are better to do in loco.


# 3.  Algorithms used and reason for that


Thinking about the problem and how I would consider the best algorithm for this problem, I have no doubt that the algorithm should be a Tree kind. This is because the way I should consider someone POI or non POI would be considering several criterias. This is the way the Tree algorithm behaves  and it would not be a clustering k-means or even a linear regression … (Although linear could be applied, but there was no clear indication if a person was POI or non POI using such method).

I created an environment test and I have tested the performance of the following algorithms:

### Using non SVM machines (Kmeans algorithm, clustering):

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=100,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=0, tol=0.0001, verbose=0)
        Accuracy: 0.80600       Precision: 0.21384      Recall: 0.17000 F1: 0.18942     F2: 0.17727
        Total predictions: 15000        True positives:  340    False positives: 1250   False negatives: 1660   True negatives: 11750




### Using Gaussian Naive bayes

 
GaussianNB(priors=None)
        Accuracy: 0.33673       Precision: 0.15693      Recall: 0.90900 F1: 0.26765     F2: 0.46413
        Total predictions: 15000        True positives: 1818    False positives: 9767   False negatives:  182   True negatives: 3233



### Using Decision tree

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
        Accuracy: 0.80620       Precision: 0.26588      Recall: 0.25750 F1: 0.26162     F2: 0.25913
        Total predictions: 15000        True positives:  515    False positives: 1422   False negatives: 1485   True negatives: 11578





### Using Random forest 

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=16, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
        Accuracy: 0.86167       Precision: 0.41379      Recall: 0.09000 F1: 0.14784     F2: 0.10670
        Total predictions: 15000        True positives:  180    False positives:  255   False negatives: 1820   True negatives: 12745



### Using Adaboost combined with Random Forest

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
        
 The time processing was considered also as indicator of performance between the algorithms and just for the last case “Adaboost combined with Random Forest” the performance was worse than the others. Off course there are 2 algorithms combined to give a better performance and it has cost more time to be processed. But I preferred to use this algorithm to have a better precision and accuracy instead  to use another one that could have a low precision.

time processing test: 354.184 s       


# 4. Tunning parameters



When I choose the Random Forest algorithm combined with Adaboost my idea was to allow the reclassification in  a non deeper level. I setup the Adaboost classifier to have 10 estimators (not more) in order to avoid delays during computational processing. For the RandomForest I have set the variable max_depth=16 in order to have also a better precision.



# 5. Validation


Validation is a way to estimate the machine learning algorithm trained performance. The method I used to validate my algorithm was based in criteria of precision , accuracy and recall. But mainly precision and accuracy. The provided tester.py script was used in order to evaluate some metrics regarding specifically the script. 


# 6. Evaluation metrics 

Using Adaboost classifier combined with Random Forest I could obtain the following results from tester.py

Accuracy: 0.86567       Precision: 0.48538 


According to the results my accuracy , considered the samples provided was about 86.7%  and the precision 48.5%. So the algorithm could predict the number of cases with a good accuracy, although the precision was medium. The fact of precision indicates almost 50% means that there are several cases 









