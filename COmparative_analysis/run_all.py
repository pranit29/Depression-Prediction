import Decissiontree as dt
import K_Nearestneighbours as kn
import NaiveBayes as nb

import SVM as svm
dt_value=dt.decision_tree()
kn_value=kn.k_nearest()
nb_value=nb.n_bayes()

svm_1,svm_2=svm.svm()
print("DT::"+str(dt_value))
print("KN::"+str(kn_value))
print("NB::"+str(nb_value))

print("SVM_Linear::"+str(svm_1))
print("SVM_RBF::"+str(svm_2))

def plot():
    import matplotlib.pyplot as plt 
   
    left = [1, 2, 3 , 4 , 5] 
  
# heights of bars 
    height = [float(dt_value),float(kn_value),float(nb_value),float(svm_1),float(svm_2)] 
  
# labels for bars 
    tick_label = ['DT', 'KNN', 'NB', 'SVM_L','SVM_R'] 
  
# plotting a bar chart 
    plt.bar(left, height, tick_label = tick_label, 
        width = 0.8, color = ['red', 'yellow', 'blue','pink','green']) 
  
# naming the x-axis 
    plt.xlabel('x - axis') 
# naming the y-axis 
    plt.ylabel('y - axis') 
# plot title 
    plt.title('Accuraccy Visualization!')
    plt.show()

plot()
