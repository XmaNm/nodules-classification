# nodules-classification
This is a repository about nodules classification
We extract 4，578 images from LIDC-IDRI dataset，then conduct data augment on that.Finally,we get 20，000 experimental images and use Denosising AutoEncoder(DAE) and ResNet-34 extract nodules features.And we extract LBP&HOG features.
reference:
dae：https://github.com/ramarlina/DenoisingAutoEncoder
resnet：https://github.com/wenxinxu/resnet-in-tensorflow
