import cv2,os
path = "5/"
length = os.listdir(path)

for ImageName in length:
    x = cv2.imread(os.path.join(path, ImageName), cv2.IMREAD_GRAYSCALE)
    x_r = cv2.resize(x,(64,64))
    cv2.imwrite("5n/" + str(ImageName),x_r)