#import all required libraries   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

def n_bayes():
#reading and analysing dataset and splitting it
        dataset=pd.read_csv(r"stressdataset.csv")
        dataset.isnull().sum()
        dataset.head()
        x=dataset.iloc[:,:-1].values
        y=dataset.iloc[:,10].values



        for index, item in enumerate(y): #so convert them to 0  or 1 
                if not (item == 0.0):       # that is either Yes or No
                        y[index] = 1

#splitting train and test dataset
        from sklearn.model_selection import train_test_split
        train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.75,random_state=0)

#scaling variables on -3 to 3 scale
        from sklearn.preprocessing import StandardScaler
        sc_x=StandardScaler()
        train_x[14:16]=sc_x.fit_transform(train_x[14:16])
        test_x=sc_x.transform(test_x)
        X_train = sc_x.fit_transform(train_x)
        X_test = sc_x.transform(test_x)

        from sklearn.naive_bayes import GaussianNB #Naive bayes
        model=GaussianNB()
        model.fit(train_x,train_y)
        prediction=model.predict(test_x)
        print('The accuracy of the NaiveBayes is',100*metrics.accuracy_score(prediction,test_y))
        return 100*metrics.accuracy_score(prediction,test_y)











