import  numpy,os,cv2,re
# from collections import Counter
path = "data/"
length = os.listdir(path)
TxtFile = open('labelqc.txt', 'rb').readlines()
strTxtFile = (" ".join(TxtFile))
total_label = numpy.zeros((len(length)))
total_data = numpy.zeros((len(length),32,32))
k = 0
# print strTxtFile
for ImageName in length:
    x = cv2.imread(os.path.join(path, ImageName), cv2.IMREAD_GRAYSCALE)
    index = ImageName
    char = ImageName[0:10]
    index = re.sub("\D", "", index)
    index = char + index
    index = index[0:15]
    total_data[k] = x
    try:
        label = strTxtFile[strTxtFile.index(index)+len(index)+1:strTxtFile.index(index)+len(index)+10]
        total_label[k] = int(label[-1])
        k += 1
    except:
        os.remove("./adjust_image/"+str(ImageName))
        print index
total_data = numpy.reshape(total_data, [total_data.shape[0], -1, ])

print total_data.shape

