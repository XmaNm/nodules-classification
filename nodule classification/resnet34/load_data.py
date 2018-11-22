#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: load_data.py
# Author: Yahui Liu <yahui.cvrs@gmail.com>

import cv2
import os
import numpy as np

DATA_LEN = 1024
CHANNEL_LEN = 1024
SHAPE = 32

def imread(im_path, shape=None, color="RGB", mode=cv2.IMREAD_UNCHANGED):
  im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

  if color == "RGB":
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im = np.transpose(im, [2, 1, 0])
  # if shape != None:
  #   assert isinstance(shape, int)
  #   im = cv2.resize(im, (shape, shape))
  return im

def read_data(filename, data_path, shape=None, color='RGB'):
  """
     filename (str): a file 
       data file is stored in such format:
         image_name  label
     data_path (str): image data folder
     return (numpy): a array of image and a array of label
  """ 
  if os.path.isdir(filename):
    print "Can't found data file!"
  else:
    f = open(filename)
    lines = f.read().splitlines()
    count = len(lines)

    data = np.zeros((count, DATA_LEN), dtype=np.uint8)
    #label = np.zeros(count, dtype=np.uint8)
    lst = [ln.split('\t')[0] for ln in lines]

    label = [ln.split('\t')[-1] for ln in lines]

    idx = 0
    s, c = SHAPE, CHANNEL_LEN

    for ln in lines:
      fname, lab = ln.split('\t')

      im = imread(os.path.join(data_path, fname), shape=s, color='GRAY')

      '''
      im = cv2.imread(os.path.join(data_path, fname), cv2.IMREAD_UNCHANGED)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, (s, s))
      '''

      data[idx] = np.reshape(im[:,:],c)

      label[idx] = int(lab[-1])

      idx = idx + 1



    return data, label, lst
