#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: demo.py
# Author: Yahui Liu <yahui.cvrs@gmail.com>

import os, cv2
from pickled import *
from load_data import *
d = os.path.dirname(__file__)
parent_path = os.path.dirname(d)

# data_path = './adjust_image'
# file_list = './adjust_image/labelqc.txt'
save_path = './data/bin'

if __name__ == '__main__':
    # data, label, lst = read_data(file_list, data_path, shape=32)
    data = np.load(parent_path + "/data_aug.npy")
    data = np.resize(data, [data.shape[0], data.shape[1] * data.shape[2]])
    label = np.load(parent_path + "/label_aug.npy")
    label = label.tolist()
    lst = np.load(parent_path + "/fname.npy")
    pickled(save_path, data, label, lst, bin_num = 6)
    os.rename(os.path.join(save_path, "data_batch_5"), os.path.join(save_path, "test_batch"))

