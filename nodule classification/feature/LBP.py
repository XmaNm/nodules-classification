#coding:utf-8
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.svm import SVR
import os
from skimage import feature as skft
import pandas as pd
d = os.path.dirname(__file__)
parent_path = os.path.dirname(d)

n = 450
data = np.load(parent_path + "/data_e.npy")
label = np.load(parent_path + "/label.npy")
indices = np.random.permutation(data.shape[0])
data = data[indices]
label = label[indices]
radius = 1
n_point = radius * 8

def makeMultiClass(y):
    u = np.unique(y)
    coords = {}
    for idx in range(len(u)):
        coords[str(u[idx])] = idx
    V = np.zeros((len(y), len(u)))
    for idx in range(len(y)):
        V[idx, coords[str(y[idx])]] = 1
    return V



def texture_detect(train_data,test_data):
    train_hist = np.zeros( (train_data.shape[0],256) )
    test_hist = np.zeros( (test_data.shape[0],256) )
    for i in np.arange(train_data.shape[0]):
        lbp=skft.local_binary_pattern(train_data[i],n_point,radius,'default')
        max_bins = int(lbp.max() + 1)
        train_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    for i in np.arange(test_data.shape[0]):
        lbp = skft.local_binary_pattern(test_data[i],n_point,radius,'default')
        max_bins = int(lbp.max() + 1)
        test_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))

    return train_hist,test_hist


train_data = data[:n]
test_data = data[n:]
train_label = label[:n]
test_label = label[n:]

train_hist, test_hist = texture_detect(train_data,test_data)

svr_rbf = SVR(kernel='rbf', C=1.0, gamma=0.1, max_iter=5000)
acc = OneVsRestClassifier(svr_rbf,-1).fit(train_hist, train_label).score(test_hist,test_label)
# acc1 = OneVsOneClassifier(svr_rbf,-1).fit(train_hist, train_label).score(test_hist,test_label)
# clf = OneVsRestClassifier(svr_rbf,-1).fit(train_hist, train_label)
prey = svr_rbf.fit(train_hist,train_label).predict(test_hist)
# print prey
label = [1,2,3,4,5]
prey = pd.cut(prey,5,labels=label)
prey = np.asarray(prey)
sumr = 0
sum = 0
for j in range(test_label.shape[0]):
    # print prey[j], test_label[j]
    sum += prey[j]==test_label[j]
print "accuracy: %2.2f%%"%(100*sum/test_label.shape[0])
# print sum,sumr


# x = np.arange(1,501)
# # print x.shape,prey.shape
# fig = plt.figure()
# plt.scatter(x,prey)
# plt.scatter(x,test_label,marker='x')
# plt.show()