import cv2,os
import numpy as np

def contrast_brightness_image(src1, a, g):
    h, w= src1.shape
    src2 = np.zeros([h, w], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)
    return dst

def loadPicture(path,label):
    length = os.listdir(path)
    total_label = np.zeros((len(length)))
    total_data = np.zeros((len(length),64,64))
    k = 0
    for ImageName in length:
        x = cv2.imread(os.path.join(path, ImageName), cv2.IMREAD_GRAYSCALE)
        x = contrast_brightness_image(x, 1.5, 1.0)
        total_data[k] = x
        total_label[k] = label
        k +=1
    return total_data,total_label

# data1,label1 = loadPicture("data/1_aug/",1)
# data2,label2 = loadPicture("data/2_aug/",2)
# data = np.concatenate([data1,data2])
# label = np.concatenate([label1,label2])
# data3,label3 = loadPicture("data/3_aug/",3)
# data = np.concatenate([data,data3])
# label = np.concatenate([label,label3])
# data4,label4 = loadPicture("data/4_aug/",4)
# data = np.concatenate([data,data4])
# label = np.concatenate([label,label4])
# data5,label5 = loadPicture("data/5_aug/",5)
# data = np.concatenate([data,data5])
# label = np.concatenate([label,label5])
# np.save('data_aug.npy',data)
# np.save('label_aug.npy',label)
data1,label1 = loadPicture("1/",1)
data2,label2 = loadPicture("2/",2)
data = np.concatenate([data1,data2])
label = np.concatenate([label1,label2])
data3,label3 = loadPicture("3/",3)
data = np.concatenate([data,data3])
label = np.concatenate([label,label3])
data4,label4 = loadPicture("4/",4)
data = np.concatenate([data,data4])
label = np.concatenate([label,label4])
data5,label5 = loadPicture("5/",5)
data = np.concatenate([data,data5])
label = np.concatenate([label,label5])
np.save('data_e.npy',data)
np.save('label.npy',label)