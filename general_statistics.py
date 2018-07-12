#!/usr/bin/python
import sys
import pickle
import numpy as np
import pickle


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
     data_dict = pickle.load(data_file)


my_dataset = data_dict
cnt_false = 0
cnt_true  = 0
cnt       = 0


for data_point in my_dataset.values():
  if data_point['poi']==False:
    cnt_false+=1
  elif data_point['poi']==True:
    cnt_true+=1
  else:
    cnt+=1

def check_variable(var):
  cnt_na  = 0 
  cnt_info= 0
  out =[]
  total = 0
  for data_point in my_dataset.values():
     if data_point[var]=="NaN":
       cnt_na+=1
     else:
       cnt_info+=1
  total = cnt_na + cnt_info
  percent_info = 100* (cnt_info/float(total))    
  percent_info = round(percent_info, 2)
  out=[cnt_info,cnt_na, percent_info,total]
  return out    
    
print "                      Dataset Size:", len(my_dataset)
print " Number of POI=true in the dataset:",  cnt_true
print "Number of POI=false in the dataset:" , cnt_false
print "                            Others:" , cnt

print " -------------------- Statistics about financial variables -----------------------------------------------------" 


financial_list = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] # You will need to use more features
printer =""
for items in financial_list:
  saida = check_variable(items)
  offset = len("restricted_stock_deferred") - len(items)
  printer += ' ' * offset
  print "Financial variable", items, ":",printer, "% of info provided", saida[2], "%", "(Data informed:",saida[0],"Not Informed:", saida[1], " Total:",saida[3],")" 
  printer=""     
