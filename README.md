# nodules-classification
This is a repository about nodules classification
We extract 4,578 images from LIDC-IDRI dataset，then conduct data augment on that.Finally,we get 20,000 experimental images and use Denosising AutoEncoder(DAE) and ResNet-34 extract nodules features.And we extract LBP&HOG features.  

## Introduction
Early identification and classification of pulmonary nodules are essential for improving lung cancer survival rate and are considered the key processes for computer-assisted diagnosis. To address this topic, the present study proposed a method for predicting the malignant phenotype of pulmonary nodules based on weighted voting rules. This method used the pulmonary nodule regions of interest as the input and extracted the features of the pulmonary nodules by the Denoising Auto Encoder, ResNet-18. Moreover, it modified the texture and shape features (SF) to assess the malignant phenotype of the pulmonary nodules. Based on their classification accuracy (Acc), the different classifiers were assigned to different weights. Finally, an integrated classifier was obtained to score the malignant phenotype of the pulmonary nodules. The present study included experiments conducted by extracting the corresponding lung nodule image data from the Lung Image Database Consortium-Image Database Resource Initiative (LIDC-IDRI).

## reference:  
dae：https://github.com/ramarlina/DenoisingAutoEncoder  
resnet：https://github.com/wenxinxu/resnet-in-tensorflow  
