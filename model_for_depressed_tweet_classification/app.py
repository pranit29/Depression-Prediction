from flask import Flask, render_template, request, redirect, url_for, flash
from flask.globals import session
import pickle , re, os
import pandas as pd
import time 
from csv import reader
print("Enter the text to predict:")
text=input()
filename= 'Model4_MultinomialNB.sav'
loaded_model = pickle.load(open(filename, 'rb'))
tf = pickle.load(open("feature.pkl", 'rb'))
ResultAnswer = loaded_model.predict(tf.transform([text]).toarray())
            
if ResultAnswer[0] == 1:
   
    
    print("Depreesed Tweet detected...")
else:
    
    print("Normal Tweet detected..")
            
time.sleep(2000)