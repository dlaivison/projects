#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

#features_list = ['poi','from_this_person_to_poi','from_poi_to_this_person','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] # You will need to use more features

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] # You will need to use more features


#features_list = ['poi','salary']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Before to remove the outliers I will plot the chart in order to better identify 
# Identified the key "TOTAL" as main outlier
# Removing the user "TOTAL" and major POI already identified persons: Kenneth Lay, Jeffey Skilling
# Identified another big value but it was not discharged due the fact to be an important data 

data_dict.pop("TOTAL",0)
data_dict.pop("LAY KENNETH L",0)
data_dict.pop("SKILLING JEFFREY K",0)
data_dict.pop("FREVERT MARK A",0)
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Identifying Outliers
#for elem in my_dataset:
#  if (my_dataset[elem]["salary"] > 1000000)and (my_dataset[elem]["salary"]!="NaN"):
#    print elem,my_dataset[elem]["salary"]

### Extract features and labels from dataset for local testing
data      = featureFormat(my_dataset, features_list, sort_keys = True)


#Creating the new feature  net_worth (Sum of all payments received)

out= np.array([])
#print data[0]
for elem in data:
  net_worth =  elem[1]+elem[2]+elem[3]+elem[4]+elem[5]+elem[6]+ elem[7]+elem[8]+elem[9]+elem[10]+elem[11]+elem[12]+elem[13]+elem[14]
  out= np.append(out,net_worth)

c = np.hstack((data,np.atleast_2d(out).T))

index =0 
for point in data:
    total_stock_value = point[8]
    net_worth = point[1]+point[2]+point[3]+point[4]+point[5]+point[6]+point[7]+point[8]+point[9]+point[10]+point[11]+point[12]+point[13]+point[14]
    index +=1
    if net_worth > 100000000:
      print net_worth, point[1], point[2],index, data_dict.keys()[index]
    plt.scatter(total_stock_value, net_worth )

plt.xlabel("Total Stock Value")
plt.ylabel("net_worth")
plt.show()

labels, features = targetFeatureSplit(data)

#print labels
#print features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

#clf = RandomForestClassifier(max_depth=2, random_state=0)
#clf  = AdaBoostClassifier(RandomForestClassifier(max_depth=2),n_estimators=40,learning_rate=1)
#clf = GaussianNB()
clf = linear_model.LinearRegression()
#clf = tree.DecisionTreeClassifier()

clf.fit(features,labels)




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print clf.score(features_test, labels_test)

test_classifier(clf, my_dataset,features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
