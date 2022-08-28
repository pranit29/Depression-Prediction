from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import pickle
import time
import json
import numpy as np

app = Flask(__name__)
import base64
import os
import pickle
import sklearn
import json
import depressed_tweet_classification as stc
@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return "Hello Boss3!"
import real_time_video as rtv
@app.route('/predictStressLevel', methods=['POST'])
#@app.route('/predictStressLevel')
def predictStressLevel():
        my_array_list = list()
  
        agnry_status=request.form['ans1']
        print(agnry_status)
        neg_feeling=request.form['ans2']
        unpleasantness=request.form['ans3']
        grumpy_mood=request.form['ans4']
        anxiousness=request.form['ans5']
        math=request.form['correct']
        usage=request.form['appUsage']
        face_mood=request.form['imageData']
        text_=request.form['text']
        
       
        imgdata = base64.b64decode(face_mood)
        filename = 'output.jpg'  
        with open(filename, 'wb') as f:
            f.write(imgdata)
        tweet_result=stc.classifyTest(text_)
        face_mood=rtv.detect_emotion()
        my_array_list.append(agnry_status)
        my_array_list.append(neg_feeling)
        my_array_list.append(unpleasantness)
        my_array_list.append(grumpy_mood)
        my_array_list.append(anxiousness)
        my_array_list.append(math)
        my_array_list.append(usage)
        my_array_list.append(face_mood)
        my_array_list.append(72)
        my_array_list.append(34)
        print(my_array_list)
        
      
        with open("pp.pkl", 'rb') as file:
          pickle_model = pickle.load(file)
          print(my_array_list)
          output=pickle_model.predict([my_array_list]) 
          print(output)
          print(output[0])
          text_classification=""
          if tweet_result==1:
              text_classification=" Stressed text detected."
          else:
              text_classification="No stressed text detected. "
              
          x={"success":"Text Classification: "+text_classification+" \n Stress level detected is: Level "+str(output[0])}
            #return render_template('index.html', value=output)
          return json.dumps(x)
   
            
        
        x={"success":str("received")}
            
        return json.dumps(x)
    

  


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)

time.sleep(2000)
