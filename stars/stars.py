import random
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import tensorflow.contrib.learn as skflow
import numpy as np

data = pandas.read_csv('hygdata_v3.csv')
# id
# hip
# hd
# hr
# gl
# bf
# proper
# ra
# dec
# dist
# pmra
# pmdec
# rv
# mag
# absmag
# spect
# ci
# x
# y
# z
# vx
# vy
# vz
# rarad
# decrad
# pmrarad
# pmdecrad
# bayer
# flam
# con
# comp
# comp_primary
# base
# lum
# var
# var_min
# var_max
unwanted_data = []
data = data.query('dist != 100000.0000')

y, X = data['dist'], data[['absmag','mag']]
y = pandas.cut(y,10,labels=range(0,10))


alphas2 = np.arange(0.01,1,0.050)
learning_rates = alphas2
batch_sizes = [8,16,32,64,128,256]


best_score = 0.01
best_rate = 0
best_b_size = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)



for learning_rate in learning_rates:

    for batch_size in batch_sizes:

        random.seed(42)
        tflr = skflow.TensorFlowLinearClassifier(n_classes=10, batch_size=batch_size,
                                                 steps=5000, learning_rate=learning_rate)
        tflr.fit(X_train, y_train)
        score = accuracy_score(tflr.predict(X_test), y_test)
        if best_score < score:
            best_score = score
            best_rate = learning_rate
            best_b_size = batch_size
            print "New best accuracy: "+str(score)+" / learning_rate: "+str(learning_rate)+" / batch_size: "+str(batch_size)
