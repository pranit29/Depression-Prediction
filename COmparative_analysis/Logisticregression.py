import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')


dataset=pd.read_csv(r"stressdataset.csv")
dataset.isnull().sum()
dataset.head()
from sklearn.preprocessing import Imputer

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,10].values
'''
imputer=Imputer(missing_values='?',strategy='mean',axis=0)
imputer=imputer.fit(x[:,:-1])
x[:,:-1]=imputer.transform(x[:,:-1])
'''
data = pd.read_csv('stressdataset.csv')
X = data.iloc[:, [2, 3]].values
Y = data.iloc[:, 4].values



 
for index, item in enumerate(y):   # Last row gives 4 diff types of output , so convert them to 0  or 1 
	if not (item == 0.0):       # that is either Yes or No
		y[index] = 1

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.75,random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


f,ax=plt.subplots(1,2,figsize=(18,8))
dataset['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Stressed')
ax[0].set_ylabel('')
sns.countplot('target',data=data,ax=ax[1])
ax[1].set_title('target')
plt.show()



#Correlation Matrix
sns.heatmap(dataset.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()



from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
train_x[14:16]=sc_x.fit_transform(train_x[14:16])
test_x=sc_x.transform(test_x)
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(train_x,train_y)
accuracy = 0.38
pred_y=classifier.predict(test_x)

from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)

classifier_lr.fit(X_train, Y_train)
Y_pred = classifier_lr.predict(X_test)


import pickle

with open('svc.pkl','wb') as m:
        pickle.dump(classifier_lr,m)

from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(pred_y,test_y)
from sklearn.metrics import classification_report
print(classification_report(test_y, pred_y))
print('Accuracy for logistic_regression is ',100*(metrics.accuracy_score(pred_y,test_y)))


