from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import os
Datagen =ImageDataGenerator(rotation_range=90,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')
path = "5/"
length = os.listdir(path)
for ImageName in length:
    img = load_img(path + ImageName)
    x_img = img_to_array(img)
    # x_img = numpy.reshape(x_img,[x_img.shape[0],x_img.shape[1]])
    x_img = x_img.reshape((1,) + x_img.shape)
    i = 0
    for img_batch in Datagen.flow(x_img,
                              batch_size=1,
                              save_to_dir="5new/",
                              save_prefix= str(ImageName[0:15]),
                              save_format='jpg'):
        i += 1
        if i>3:
            break