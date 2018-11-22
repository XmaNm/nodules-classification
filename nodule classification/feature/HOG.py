#coding:utf-8

import numpy as np
from skimage import feature as ft
import os
from sklearn.svm import SVR
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd

d = os.path.dirname(__file__)
parent_path = os.path.dirname(d)

n = 4000
data = np.load(parent_path + "/data_e.npy")
label = np.load(parent_path + "/label.npy")
indices = np.random.permutation(data.shape[0])
data = data[indices]
label = label[indices]

def hogextract(image):
    feature = np.zeros((image.shape[0],324))
    for i in range(image.shape[0]):
        feature[i] = ft.hog(image[i])
    return feature

train_data = data[:n]
test_data = data[n:]
train_label = label[:n]
test_label = label[n:]

train_datah = hogextract(train_data)
test_datah = hogextract(test_data)
svr_rbf = SVR(kernel='linear', C=1.0, gamma=0.1)
#
clf = OneVsRestClassifier(svr_rbf,-1).fit(train_datah, train_label)
prey = svr_rbf.fit(train_datah,train_label).predict(test_datah)
acc = OneVsRestClassifier(svr_rbf,-1).fit(train_datah, train_label).score(test_datah,test_label)
label = [1,2,3,4,5]
prey = pd.cut(prey,5,labels=label)
prey = np.asarray(prey)
sumr = 0
sum = 0
for j in range(test_label.shape[0]):
    # print prey[j], test_label[j]
    sum += prey[j]==test_label[j]
print "accuracy: %2.2f%%"%(100*sum/test_label.shape[0])
#     sumr += abs(round(prey[j])-test_label[j])
#     sum += abs(prey[j]) - test_label[j]
# print sum,sumr