#import all required libraries   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
#reading and analysing dataset and splitting it
def svm():
        dataset=pd.read_csv(r"stressdataset.csv")
        dataset.isnull().sum()
        dataset.head()
        x=dataset.iloc[:,:-1].values
        y=dataset.iloc[:,10].values


#splitting train and test dataset
        from sklearn.model_selection import train_test_split
        train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.75,random_state=0)

#scaling variables on -3 to 3 scale
        from sklearn.preprocessing import StandardScaler
        sc_x=StandardScaler()
        accuracy = 0.35
        train_x[14:16]=sc_x.fit_transform(train_x[14:16])
        test_x=sc_x.transform(test_x)
        X_train = sc_x.fit_transform(train_x)
        X_test = sc_x.transform(test_x)

#svm-linear kernel
        from sklearn import svm #support vector Machine
        from sklearn import metrics
        model=svm.SVC(kernel='linear',C=0.05,gamma=0.1)
        model.fit(train_x,train_y)
        prediction1=model.predict(test_x)
#here to save model

        import pickle

        with open('pp.pkl','wb') as m:
                pickle.dump(model,m)

        #print('Accuracy for linear SVM is ',100*(metrics.accuracy_score(prediction1,test_y)))

        model1=svm.SVC(kernel='rbf',C=0.05,gamma=0.1)
        model1.fit(X_train,train_y)
        Y_pred=model1.predict(test_x)

#svm-rbf kernel
#if gamma = 0.1 we are getting 95% accuracy but it is overfitting
        from sklearn import svm #support vector Machine
        model=svm.SVC(kernel='rbf',C=1,gamma=0.2)
        model.fit(train_x,train_y)
        prediction1=model.predict(test_x)
       # print(test_y)

        print('Accuracy for rbf SVM is ',100*(metrics.accuracy_score(prediction1,test_y)))
        return 100*(metrics.accuracy_score(prediction1,test_y)),100*(metrics.accuracy_score(prediction1,test_y))
svm()
