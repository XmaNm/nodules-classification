# import cv2,os
# path = "1/"
# length = os.listdir(path)
#
# def rotate(image, angle, center=None, scale=1.0):
#     (h, w) = image.shape[:2]
#     if center is None:
#         center = (w / 2, h / 2)
#         M = cv2.getRotationMatrix2D(center, angle, scale)
#         rotated = cv2.warpAffine(image, M, (w, h))
#     return rotated
#
# for ImageName in length:
#     x = cv2.imread(os.path.join(path, ImageName), cv2.IMREAD_GRAYSCALE)
#     x_t = rotate(x,180)
#     cv2.imwrite(ImageName[0:-4] + "r" + ImageName[-4:],x_t)
