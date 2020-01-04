import pandas as pd
import numpy as nm

affairs = pd.read_csv("C:/My Files/Excelr/05 - Logistic Regression/Assignment/affairs.csv")
affairs.columns
affairs = affairs.drop(['Unnamed: 0','education', 'occupation', 'rating'],axis=1)

affairs['affairs'] = nm.where(affairs["affairs"] > 0 , 1 ,0)
affairs['children'] = nm.where(affairs["children"] == 'yes' , 1 ,0)
affairs['gender'] = nm.where(affairs["gender"] == 'male' , 1 ,0)
affairs.describe()

pd.set_option('display.expand_frame_repr', False)
affairs.isnull().sum()

import statsmodels.formula.api as sm
model1 = sm.logit('affairs ~ gender+age+yearsmarried+children+religiousness',data = affairs).fit()
model1.summary()
model1_pred = model1.predict(affairs)
affairs['pred_prob'] = model1_pred

affairs["aff_val"] = nm.zeros(601)
affairs.loc[model1_pred>=0.5,"aff_val"] = 1

from sklearn.metrics import classification_report
classification_report(affairs.aff_val,affairs.affairs)
confusion_matrix = pd.crosstab(affairs['affairs'],affairs.aff_val)
confusion_matrix

accuracy = (445+7)/(601)
accuracy

# ROC curve 
from sklearn import metrics
import matplotlib.pyplot as plt
fpr, tpr, threshold = metrics.roc_curve(affairs.affairs, model1_pred) 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
# area under ROC curve 
roc_auc = metrics.auc(fpr, tpr) 
roc_auc
