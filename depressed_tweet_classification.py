from flask import Flask, render_template, request, redirect, url_for, flash
from flask.globals import session
import pickle , re, os
import pandas as pd
from csv import reader
filename= 'Model4_MultinomialNB.sav'
def classifyTest(tweet):
    loaded_model = pickle.load(open(filename, 'rb'))
    tf = pickle.load(open("feature.pkl", 'rb'))

    ResultAnswer = loaded_model.predict(tf.transform([tweet]).toarray())
            
    if ResultAnswer[0] == 1:
        print("Depressed Tweet detected...")
        return 1
    else:
        print("Normal Tweet detected..")
        return 0
            
