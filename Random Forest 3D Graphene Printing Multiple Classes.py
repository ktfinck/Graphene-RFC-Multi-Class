# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:56:37 2021

@author: Kyle Finck
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
#%% Data Import and Trimming

dataset = pd.read_excel(r'ProcessParameterDataset.xlsx')
dataset.columns = ['nozzle speed', 'flowrate', 'voltage', 'resistance']
dataset = dataset.replace(to_replace='non conductive line ', value=0)

NuzzelData     = np.array(dataset['nozzle speed'], dtype=np.float64)
FlowrateData   = np.array(dataset['flowrate'], dtype=np.float64)
VoltageData    = np.array(dataset['voltage'], dtype=np.float64)
ResistanceData = np.array(dataset['resistance'], dtype=np.float64)
CombinedData   = dataset[['nozzle speed', 'flowrate', 'voltage']].to_numpy()



TrimmedData = np.delete(ResistanceData, np.where(ResistanceData == 0))
SortedData = np.sort(TrimmedData)
ResistanceMedian = np.median(TrimmedData)
HiLowData = np.zeros(np.size(ResistanceData))
HiData = np.zeros(np.size(ResistanceData))
LowData = np.zeros(np.size(ResistanceData))

for i in range(0, np.size(HiLowData)):
    if 0<ResistanceData[i]<=ResistanceMedian:
        LowData[i] = ResistanceData[i]
    elif ResistanceMedian<ResistanceData[i]:
        HiData[i] = ResistanceData[i]
        
LowData = np.delete(LowData, np.where(LowData == 0))
HiData = np.delete(HiData, np.where(HiData == 0))

ResistanceLowMedian = np.median(LowData)
ResistanceHiMedian = np.median(HiData)

for i in range(0, np.size(HiLowData)):
    if 0<ResistanceData[i]<=ResistanceLowMedian:
        HiLowData[i] = 1
    elif ResistanceLowMedian<ResistanceData[i]<=ResistanceMedian:
        HiLowData[i] = 2
    elif ResistanceMedian<ResistanceData[i]<=ResistanceHiMedian:
        HiLowData[i] = 3
    elif ResistanceHiMedian<ResistanceData[i] or ResistanceData[i] == 0:
        HiLowData[i] = -1 
        
X = CombinedData
y = HiLowData

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)


RFC = RandomForestClassifier(n_estimators=100, 
                             criterion= 'gini',
                             bootstrap=True,
                             max_depth=None,
                             max_features = None,
                             oob_score = True)
RFC.fit(X_train, y_train)

print(RFC.score(X_test,y_test))



y_pred_test = RFC.predict(X_test)
y_pred_train= RFC.predict(X_train)
y_pred = RFC.predict(X)


fig, ax = plt.subplots(num='Overall', dpi=200)
plot_confusion_matrix(RFC, X, y, cmap= 'bone', ax=ax)
plt.title('Confusion Matrix: RFC Overall')
plt.xlabel('Predicted')
plt.ylabel('Actual')

fig, ax = plt.subplots(num='Test', dpi=200)
plot_confusion_matrix(RFC, X_test, y_test, cmap= 'bone', ax=ax)
plt.title('Confusion Matrix: RFC Test')
plt.xlabel('Predicted')
plt.ylabel('Actual')

fig, ax = plt.subplots(num='Train', dpi=200)
plot_confusion_matrix(RFC, X_train, y_train, cmap= 'bone', ax=ax)
plt.title('Confusion Matrix: RFC Train')
plt.xlabel('Predicted')
plt.ylabel('Actual')


importance = RFC.feature_importances_

std = np.std([tree.feature_importances_ for tree in RFC.estimators_],axis=0)
indices = np.argsort(importance)[::-1]
label = []
for f in range(X.shape[1]):
    if indices[f] == 0:
        label.append('Nozzle Speed')
    elif indices[f] == 1:
        label.append('Flowrate')
    elif indices[f] == 2:
        label.append('Voltage')

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d.  %s (%f)" % (f + 1, label[f], importance[indices[f]]))

# Plot the impurity-based feature importances of the forest
fig, ax = plt.subplots(dpi=200)
plt.title("Feature Importance: RFC Gini Impurity Multi-Class")
plt.bar(range(X.shape[1]), importance[indices],
        color=["dodgerblue", "goldenrod", "firebrick"], yerr=std[indices], align="center", width=0.7, capsize=5)
plt.xticks(range(X.shape[1]), label)
plt.xlim([-.5, X.shape[1]-.5])
plt.ylim([0,1])
ax.yaxis.grid()
plt.ylabel('Importance')


k_fold = KFold(n_splits=5)
k_fold.get_n_splits(X)
# print(k_fold)

for train_index, test_index in k_fold.split(X,y):
      # print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      #print(X_train, X_test, y_train, y_test)


k_fold_score_RFC = cross_val_score(RFC, X, y, cv=k_fold, n_jobs=-1)

print('\n5-Fold Validation Score (RFC): ', k_fold_score_RFC)

LR = LinearRegression()
LR.fit(X,y)
LR_score = LR.score(X,y)


print('Linear Regression Score: ', LR_score)

