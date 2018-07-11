#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import time

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier



def featureScaling(arr):
    out=[]
    arr.sort()
    min = arr[0]
    max = arr[-1]

    for elem in arr:
        value= (elem -min)/float(max-min)
        out.append(value)

    return out

def check_value(val):
    if val=="NaN" :
      return 0
    else:
      return val



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### financial f['poi','salary',eatures: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Before to remove the outliers I will plot the chart in order to better identify 
# Identified the key "TOTAL" as main outlier
# Removing the data_point "TOTAL" and major POI already identified persons: Kenneth Lay, Jeffey Skilling
# Identified another big value but it was not discharged due the fact to be an important data 

data_dict.pop("TOTAL",0)
data_dict.pop("LAY KENNETH L",0)
data_dict.pop("SKILLING JEFFREY K",0)
data_dict.pop("FREVERT MARK A",0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for data_point in my_dataset.values():
    data_point['to_poi_message_ratio'] = 0
    data_point['from_poi_message_ratio'] = 0
    data_point['net_worth'] = 0
    if float(data_point['from_messages']) > 0:
        data_point['to_poi_message_ratio'] = float(data_point['from_this_person_to_poi'])/float(data_point['from_messages'])
    if float(data_point['to_messages']) > 0:
        data_point['from_poi_message_ratio'] = float(data_point['from_poi_to_this_person'])/float(data_point['to_messages'])
    data_point['net_worth']= check_value(data_point['salary']) \
                      + check_value(data_point['deferral_payments'])\
                      + check_value(data_point['total_payments'])\
                      + check_value(data_point['loan_advances'])\
                      + check_value(data_point['bonus'])\
                      + check_value(data_point['restricted_stock_deferred'])\
                      + check_value(data_point['deferred_income'])\
                      + check_value(data_point['total_stock_value'])\
                      + check_value(data_point['expenses'])\
                      + check_value(data_point['exercised_stock_options'])\
                      + check_value(data_point['other'])\
                      + check_value(data_point['long_term_incentive'])\
                      + check_value(data_point['restricted_stock'])\
                      + check_value(data_point['director_fees'])       

    
features_list.extend(['to_poi_message_ratio', 'from_poi_message_ratio','net_worth'])

#features_list = ['poi','salary','to_poi_message_ratio', 'from_poi_message_ratio','net_worth']
features_list = ['poi','other','bonus','total_stock_value','to_poi_message_ratio']


# Identifying Outliers
#for elem in my_dataset:
#  if (my_dataset[elem]["salary"] > 1000000)and (my_dataset[elem]["salary"]!="NaN"):
#    print elem,my_dataset[elem]["salary"]

### Extract features and labels from dataset for local testing
data      = featureFormat(my_dataset, features_list, sort_keys = True)
#print data

# For loop to identify the field more often
#for elem in data:
#  index=0
#  for index in range (0, len(elem)):
#    if elem[index]!=0:
#       print index


#Creating the new feature  net_worth (Sum of all payments received)
value =[]
out= np.array([])
#print data[0]
#for elem in data:
#  net_worth =  elem[2]+elem[3]+elem[4]+elem[5]+elem[6]+ elem[7]+elem[8]+elem[9]+elem[10]+elem[11]+elem[12]+elem[13]+elem[14]
#  value.append(net_worth)
#  out= np.append(out,net_worth)

#c = np.hstack((data,np.atleast_2d(out).T))
#c = np.delete(c,[1,2,3,4,5,6,7,8,9,10,11,12,13,14],axis=1)
#print c

#saida = featureScaling(value)
#print saida

index =0 
#for point in data:
#    total_stock_value  = point[8]
#    net_worth = point[2]+point[3]+point[4]+point[5]+point[6]+point[7]+point[8]+point[9]+point[10]+point[11]+point[12]+point[13]+point[14]
#    index +=1
#    if net_worth > 100000000:
#      print net_worth, point[1], point[2],index, data_dict.keys()[index]
#    plt.scatter(total_stock_value, net_worth)
#    index +=1
#    print total_stock_value

#plt.xlabel("Total Stock Value")
#plt.ylabel("net_worth")
#plt.show()

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
from sklearn import svm

#clf = RandomForestClassifier(max_depth=16, random_state=0)
#clf  = AdaBoostClassifier(RandomForestClassifier(max_depth=16),n_estimators=8,learning_rate=1)
clf  = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=32),n_estimators=16,learning_rate=1)
#clf = svm.SVC()
#clf = GaussianNB()
#clf = linear_model.LinearRegression()
#clf = tree.DecisionTreeClassifier(max_depth=16)

clf.fit(features,labels)


#### feature importances for feature selection process
feature_importances = clf.feature_importances_
sorted_idx = (-np.array(feature_importances)).argsort()
print "Rank of features"
for idx in sorted_idx:
    print "{:4f} : {}".format(feature_importances[idx], features_list[idx+1])

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

t0 = time()

test_classifier(clf, my_dataset,features_list)


print "time processing test:", round(time()-t0, 3), "s"

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
