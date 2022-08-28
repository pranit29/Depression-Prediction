from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import tensorflow as ff  #2.4.0
import keras as kk  #2.4.3
print(ff.__version__)
print(kk.__version__)

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]
import time

# starting video streaming
def detect_emotion():
    label="no"
    l_out=0
    #cv2.namedWindow('your_face')
    #camera = cv2.VideoCapture(0)
    if 1==1:
        path = r'output.jpg'
   
# Reading an image in default mode


        src = cv2.imread(path)
    #reading the frame
        #cv2.imwrite("output.jpg", frame) 
    #frame1 = imutils.resize("output.jpg",width=300)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        preds=""
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = gray.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
        
        
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
    #else: continue

 
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

        #print(label)
                #["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    
        
        #cv2.imshow('your_face', frameClone)
        #cv2.waitKey(0)
        print(label)
        if label!="no":
            
            print("Emotion Detected Successfully...")
           
            


        
            if label=="angry" or label=="sad":
                l_out=0
            elif label=="disgust" or label=="scared":
                l_out=1
            elif label=="neutral":
                l_out=2
            elif label=="happy" or label=="surprised":
                l_out=3
            
        else:
             print("Not detected...")
             return 2
            
        print(l_out)
        return l_out


