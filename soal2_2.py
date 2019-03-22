import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# CHECK PART 2
# Read a file
df = pd.read_csv('data.csv')

# Read a columns
# print(df.columns)

# add criteria as a reference
# print(df[df['Age'] <= 25])
# print(df[df['Overall'] >= 80])
# print(df[df['Potential'] >= 80])

# another reference for plotting and/or machine learning
# print(df[df['Age'] <= 25][df['Overall'] >= 80][df['Potential'] >= 80])

# variable for machine learning
rekrut = []
for x in range(len(df.values)):                # either he's recommended or not recommended
    if df['Age'][x] <= 25 and df['Overall'][x] >= 80 and df['Potential'][x] >= 80:
        rekrut.append('recommended')
    else:
        rekrut.append('notRec')

# make sure that they are synchronized
# print(len(rekrut))
# print(len(df))

# add a new column
df['Recruit'] = rekrut
print(len(df))

# PART 2
# import all the machine learning packages
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# import k-fold cross validation
from sklearn.model_selection import KFold, cross_val_score
k = KFold(n_splits=3)


# function get score

def get_score(model, xtr, ytr, xts, yts):
    model.fit(xtr,ytr)
    return model.score(xts,yts)

sLogreg = []
sSVC = []
sDT = []
sRanFor = []
sKNN = []

# Find the columns that needed to train
# print(df[['Age','Overall','Potential']])

train = df[['Age','Overall','Potential','Recruit']]
dfTrain = pd.DataFrame(train)
# print(dfTrain[['Age','Overall','Potential','Recruit']].iloc[0])


for train_index, test_index in k.split(df[['Age','Overall','Potential']]): # check line 53
    xtr = df[['Age','Overall','Potential']].iloc[train_index]
    ytr = df['Recruit'][train_index]
    
# special case for KNN
def n_values():
    n = round((len(xtr) + len(df[['Age','Overall','Potential']])) ** .5)
    if (n % 2 == 0):
        return n +1
    else:
        return n

print(cross_val_score(
    LogisticRegression(),
    xtr,
    ytr
).mean()) # 0.9558411599934091
print(cross_val_score(
    SVC(gamma='auto'),
    xtr,
    ytr
).mean()) # 0.9873125720876587
print(cross_val_score(
    tree.DecisionTreeClassifier(), 
    xtr, 
    ytr)
    .mean()) # 0.8810347668479156
print(cross_val_score(
    RandomForestClassifier(n_estimators=100),
    xtr,
    ytr
).mean()) # 0.9396935244686109
print(cross_val_score(
    KNeighborsClassifier(n_neighbors=n_values()), 
    xtr, 
    ytr)
    .mean()) # 0.9409293129016313