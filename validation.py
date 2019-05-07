# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 21:15:35 2018

@author: Erik
"""

import os, csv
from PIL import Image
from keras.models import load_model
import keras.backend as K
from keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
import glob
import pickle
import time
import requests
#from sklearn.metrics import f1_score
import scikitplot as skplt
import matplotlib.pyplot as plt

def second_opinion(id,coords,topath):
    GOOGLE_MAPS_API_URL = 'https://maps.googleapis.com/maps/api/staticmap'

    coords = str(coords[1]) + "," + str(coords[0])
    
    params = {
            'key': API_KEY,
            'size': '512x512',
            'maptype': 'satellite',
            'zoom': '20',
            'format': 'jpg',
            'center': coords
        }
        #print(params['center'])
    response = requests.get(GOOGLE_MAPS_API_URL, params=params)
    print(response.status_code)
    time.sleep(1)
        
    filepath = topath + '_gmaps.jpg'
    print(filepath)
            
    with open(filepath, 'wb') as f:
        f.write(response.content)
            


y_pred = []
y_pred_prob = []
y_true = [] 
batch_size = 32
dirs = TESTING_DIRS
modelpath = MODEL_PATH

model = load_model(modelpath)
print("Model loaded")

#load pickle of property centroids
ay_dict = pickle.load(open("ay_point_wgs.p", "rb" ))

csv_out=open("validation.csv",'w', newline='')
w=csv.writer(csv_out)

for idx, (dir) in enumerate(dirs):

    x = np.empty((0,3), dtype=float, order='C')
    
    files = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(dir + '*.jpg')]

    for i in range(0,len(files),batch_size):
        print(i)
        if i + batch_size < len(files):
            filelist = files[i:i+batch_size]
        else:
            filelist = files[i:]
        im = np.array([np.asarray(Image.open(dir+fname+'.jpg'), dtype=K.floatx()) for fname in filelist])
        im = preprocess_input(im)
        
        print(f"Predicting files {i} to {i+len(filelist)} in {dir}")
        pred = model.predict(im, batch_size=batch_size, verbose=1, steps=None)
        y_pred_prob.extend(pred[:,0])
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        flat_pred = [item for sublist in pred for item in sublist]
        y_pred.extend(flat_pred)
        if idx == 0:
            truth = np.zeros(len(filelist)).tolist()
            y_true.extend(truth)
            list_to_write = [filelist,flat_pred,truth]
        elif idx == 1:
            truth = np.ones(len(filelist)).tolist()
            y_true.extend(truth)
            list_to_write = [filelist,flat_pred,truth]
        x = np.vstack((x,np.array(list_to_write).T))
        for row in np.array(list_to_write).T:
            print(row)
            w.writerow(row)
    
    #save false predictions
    errors = x[x[:, 1] != x[:,2]]

    for row in errors:
        id_dict = row[0][:36]
        #second_opinion(id, ay_dict.get(id_dict), topath) 
        

y_pred_prob = np.array(y_pred_prob)

fpr, tpr, thresholds = skplt.metrics.roc_curve(y_true, 1-y_pred_prob)
precision, recall, thresholds = skplt.metrics.precision_recall_curve(y_true, 1-y_pred_prob)
roc_auc =  skplt.metrics.auc(fpr, tpr)
prc_auc = skplt.metrics.auc(recall, precision)

plt.figure()
plt.plot(recall, precision, color='darkorange', lw=1.5, label='Area under curve = %0.3f' % prc_auc)
plt.plot([0, 1], [0.136, 0.136], color='navy', lw=1.5, linestyle='--')
plt.xlim([0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (TPR)')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.savefig('figures/PR_curve.png')
plt.show()

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'Area under curve = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.ylabel('Recall (TPR)')
plt.xlabel('False Positive Rate')
plt.savefig('figures/ROC_curve.png')
plt.show()

pred_labels = np.asarray(y_pred)
true_labels = np.asarray(y_true)

# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
TP = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
 
# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
 
# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
 
# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    
p = TP / (TP + FP)
r = TP / (TP + FN)
f = 2*(p*r)/(p+r)

acc = (TP + TN)/(TP + TN + FP + FN)
TNR = TN/(TN+FP)
TPR = TP/(TP+FN)

bacc = (TNR+TPR)/2

csv_out.close()
