import pandas as pd
import numpy as np

#train_data['native-country'] = train_data['native-country'].fillna(train_data['native-country'].mode())


train_data = pd.read_csv('training_data.csv') # [32561 rows x 15 columns]
test_data = pd.read_csv ('testing_data.csv')  # [16281 rows x 15 columns]

#train_data = train_data.drop('race',axis=1)
#train_data = train_data.drop ('gender' , axis =1)
#train_data = train_data.drop ('marital-status' , axis = 1)
#train_data = train_data.drop ('relationship' , axis = 1)

#test_data = test_data.drop ('relationship' , axis = 1)
#test_data = test_data.drop('race',axis=1)
#test_data = test_data.drop ('gender' , axis =1) 
#test_data = test_data.drop ('marital-status' , axis = 1)

#train_data.fillna(train_data.iloc[:,13].mode(),inplace = True , axis = 1)

train_data = train_data.dropna() # train_data -> [30162 rows x 11 columns]
test_data = test_data.dropna()   # test_data -> [15060 rows x 11 columns]

  
'''v = train_data.iloc[: ,1]
v = list(v)      
v = list(set(v))

d= {}
g= 0
for i in range (len(v)):
    d[v[i]] = g
    g+=1
    
for i in range (len(train_data)):
    train_data.iloc [i , 1] = d[train_data.iloc [i , 1]]
        
v = train_data.iloc[: ,3]
v = list(v)      
v = list(set(v))'''

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data.iloc[: , 1] =le.fit_transform(train_data['workclass'])
train_data.iloc[: , 3] =le.fit_transform(train_data['education'])
train_data.iloc[: , 5] =le.fit_transform(train_data['marital-status'])
train_data.iloc[: , 6] =le.fit_transform(train_data['occupation'])
train_data.iloc[: , 9] =le.fit_transform(train_data['gender'])
train_data.iloc[: , 8] =le.fit_transform(train_data['race'])
train_data.iloc[: , 7] =le.fit_transform(train_data['relationship'])
train_data.iloc[: , 13] =le.fit_transform(train_data['native-country'])
train_data.iloc[: , 14] =le.fit_transform(train_data['income'])


test_data.iloc[: , 1] =le.fit_transform(test_data['workclass'])
test_data.iloc[: , 3] =le.fit_transform(test_data['education'])
test_data.iloc[: , 5] =le.fit_transform(test_data['marital-status'])
test_data.iloc[: , 6] =le.fit_transform(test_data['occupation'])
test_data.iloc[: , 8] =le.fit_transform(test_data['race'])
test_data.iloc[: , 9] =le.fit_transform(test_data['gender'])
test_data.iloc[: , 7] =le.fit_transform(test_data['relationship'])
test_data.iloc[: , 13] =le.fit_transform(test_data['native-country'])
test_data.iloc[: , 14] =le.fit_transform(test_data['income'])



X_train =train_data.iloc[:, :-1]
X_test = test_data.iloc [:, :-1]
y_train = train_data.iloc [: , 14]
y_test = test_data.iloc [: , 14]


from sklearn.ensemble import RandomForestClassifier
lr=RandomForestClassifier()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_RandomForest = confusion_matrix(y_test, y_pred)
# 84.3%

from sklearn.tree import DecisionTreeClassifier
lr = DecisionTreeClassifier()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_DecisionTree = confusion_matrix(y_test, y_pred)
# 80.3%

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_LogisticRegression = confusion_matrix(y_test, y_pred)
# 78.4%


from sklearn.neighbors import KNeighborsClassifier  
lr=KNeighborsClassifier()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_KNeighborsClassifier = confusion_matrix(y_test, y_pred)
# 76.7%

from sklearn.naive_bayes import GaussianNB
lr=GaussianNB()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_GaussianNB = confusion_matrix(y_test, y_pred)
# 78.8%

'''from fancyimpute import KNN    
# Use 10 nearest rows which have a feature to fill in each row's missing features
knnOutput = KNN(k=10).complete(train_data)'''

'''# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X_train.values[:, 1] = labelencoder_X_1.fit_transform(X_train.values[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()

labelencoder_X_3 = LabelEncoder()
X_train.values[:, 3] = labelencoder_X_3.fit_transform(X_train.values[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X_train = onehotencoder.fit_transform(X_train).toarray()

labelencoder_X_5 = LabelEncoder()
X_train.values[:, 5] = labelencoder_X_5.fit_transform(X_train.values[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [5])
X_train = onehotencoder.fit_transform(X_train).toarray()
'''



 
