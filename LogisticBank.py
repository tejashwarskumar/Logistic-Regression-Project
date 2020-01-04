import pandas as pd
import numpy as nm

bank = pd.read_csv("C:/My Files/Excelr/05 - Logistic Regression/Assignment/bank-full.csv")
bank.columns

bank["poutcome"].value_counts()
bank.loc[bank.pdays == -1 , 'pdays'] = 0
bank["job"].replace('admin.','admin', inplace=True)
cleanup_nums = {"job":{"blue-collar": 1, "management": 2,"technician":3,
                       "admin":4,"services":5,"retired":6,'self-employed':7,
                       "entrepreneur":8,"unemployed":9,"housemaid":10,"student":11,
                       "unknown":12},
                }

cleanup_nums_2 = {"marital":{"married": 0, "single": 1,"divorced":2},
                  "education":{"secondary": 0, "tertiary": 1,"primary":2,'unknown':3},
                  "default":{"no": 0, "yes": 1},
                  "housing":{"no": 0, "yes": 1},
                  "y":{"no":0,"yes":1},
                  "loan":{"no": 0, "yes": 1},
                  "contact":{"cellular": 0, "unknown": 1,"telephone":2},
                  "month":{"may": 0, "jul": 1,"aug":2,"jun":3,"nov":4,"apr":5,"feb":6,"jan":7,"oct":8,"sep":9,"mar":10,"dec":11},
                  "poutcome":{"unknown":0,"failure":1,"other":2,"success":4}
                  }
              
cleanup_3 = {"y":{"no":0,"yes":1}}

bank.replace(cleanup_nums, inplace=True)
bank.replace(cleanup_nums_2, inplace=True)
bank.replace(cleanup_3, inplace=True)
bank.describe()

import statsmodels.formula.api as sm
model1 = sm.logit('y ~ age+job+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome',data=bank).fit()
model1.summary()
model1_pred = model1.predict(bank)
bank["pred_val"] = model1_pred
bank["y_val"] = nm.zeros(45211)
bank.loc[model1_pred >= 0.5, "y_val"] = 1

from sklearn.metrics import classification_report
classification_report(bank.y_val,bank.y)

confusion_matrix = pd.crosstab(bank.y,bank.y_val)
confusion_matrix
accuracy = (38999+1681)/(45211) 
accuracy

from sklearn import metrics
import matplotlib.pyplot as plt

fpr, tpr, threshold = metrics.roc_curve(bank.y, model1_pred)
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Postive")

roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
roc_auc
