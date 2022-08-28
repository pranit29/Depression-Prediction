
import numpy as np  #process value in array
import matplotlib.pyplot as plt  #draw graphs
import pandas as pd   #its related to OS operation, read file etc. 
def decision_tree():
        dataset = pd.read_csv(r"stressdataset.csv")
        x = dataset.iloc[:,0:9].values  #[: rowa,0:13 columns]
        y = dataset.iloc[:, 10].values

        for index, item in enumerate(y):   # so convert them to 0  or 1 
                if not (item == 0.0):       # that is either Yes or No
                        y[index] = 1
#Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0) #75% for training , 25%for testing

# Fitting Decision Tree Classification to the Training set
        from sklearn.tree import DecisionTreeClassifier   #decision tree classifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(x_train, y_train)  #75 % input parameters x_train, 75 y_train% output of that data
# Predicting the Test set results
        y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
        from sklearn import metrics
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        from sklearn.metrics import classification_report
        print(classification_report(y_test,y_pred))
        print('The accuracy of the Decision Tree is',100*metrics.accuracy_score(y_pred,y_test)) 
        return 100*metrics.accuracy_score(y_pred,y_test)


decision_tree()
