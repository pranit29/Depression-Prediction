# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def k_nearest():
# Importing the dataset
        dataset = pd.read_csv(r"stressdataset.csv")
        x = dataset.iloc[:,0:9].values
        y = dataset.iloc[:, 10].values


        for index, item in enumerate(y):   #  so convert them to 0  or 1 
                if not (item == 0.0):       # that is either Yes or No
                        y[index] = 1
# Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)


# Feature Scaling , Normalization of data
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

# Fitting K-NN to the Training set
#from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 12, metric = 'minkowski', p = 2)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)


# Making the Confusion Matrix
        from sklearn import metrics
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        from sklearn.metrics import classification_report
        print(classification_report(y_test,y_pred))
        print('Accuracy for K-Nearest Neighbours is ',100*(metrics.accuracy_score(y_pred,y_test)))

#plotting accuracy vs k_nearest neighbours
        a_index=list(range(1,11))
        a=pd.Series()
        x=[0,1,2,3,4,5,6,7,8,9,10]
        for i in list(range(1,11)):
            model=KNeighborsClassifier(n_neighbors=i, metric = 'minkowski', p = 2) 
            model.fit(x_train,y_train)
            prediction=model.predict(x_test)
            a=a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))
        plt.plot(a_index, a)
        plt.xticks(x)
        fig=plt.gcf()
        fig.set_size_inches(12,6)
        plt.show()
        print('Accuracies for different values of n are:',a.values,'with the max value as ',100*a.values.max())
        return 100*a.values.max()

