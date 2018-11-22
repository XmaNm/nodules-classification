import cv2
import numpy as np

def contrast_brightness_image(src1, a, g):
    h, w= src1.shape

    src2 = np.zeros([h, w], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)
    return dst

src = cv2.imread("data/1/LIDC-IDRI-00521.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("src",src)
dst = contrast_brightness_image(src,1.5,1.0)
cv2.imshow("dst",dst)
cv2.waitKey(0)

