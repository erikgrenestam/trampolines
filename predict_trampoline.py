# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 21:15:35 2018

@author: Erik
"""

import csv
from PIL import Image
from keras.models import load_model
from keras.applications.inception_resnet_v2 import preprocess_input
import keras.backend as K
import numpy as np

out_dir = PATH_TO_OUTPUT
model_path = PATH_TO_MODEL
p_thresh = 0.5

batch_size = 36
model = load_model(model_path)
print("Model loaded")

for year in range(2006,2018,1):
    dir = f'C:/img/{year}/'
    csv_out=open(out_dir + f"t_pred_{year}_v3.csv",'w', newline='')
    w=csv.writer(csv_out)
    
    with open(out_dir + f"t_list_{year}.csv",'r') as mat_list:
        reader = csv.reader(mat_list)
        pred_list = list(reader)
        files = [row[1] for row in pred_list]
        files.pop(0) #remove header

    y_pred = []
    x = np.empty((0,3), dtype=float, order='C')
    
    for i in range(0,len(files),batch_size):
        print(i)
        if i + batch_size < len(files):
            filelist = files[i:i+batch_size]
        else:
            filelist = files[i:]
        namelist = [f[:36] for f in filelist]
        im = np.array([np.asarray(Image.open(dir+'img/'+fname+'.jpg'), dtype=K.floatx()) for fname in filelist])
        im = preprocess_input(im)
        print(f"Predicting files {i} to {i+len(filelist)} in {dir}")
        pred = model.predict(im, batch_size=batch_size, verbose=1, steps=None)
        pred[pred > p_thresh] = 1
        pred[pred <= p_thresh] = 0
        flat_pred = [item for sublist in pred for item in sublist]
        y_pred.extend(flat_pred)

        list_to_write = [namelist,filelist,flat_pred]
        x = np.vstack((x,np.array(list_to_write).T))
        for row in np.array(list_to_write).T:
            print(row)
            w.writerow(row)

    csv_out.close()