import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
from PIL import Image

def rgb2rgbd(rgb):
    rgb2xyz = np.array([[0.4124,0.3756,0.1805],[0.2126,0.7152,0.0722],[0.0192,0.1192,0.9505]])
    xyz2xyzd = np.array([[0.1884,0.6597,0.1016],[0.2318,0.8116,-0.0290],[0,0,1]])
    xyz2rgb = np.linalg.inv(rgb2xyz)
    M1 = np.matmul(xyz2rgb,xyz2xyzd)
    M2 = np.matmul(M1,rgb2xyz)
    rgb_reshaped = rgb.reshape((rgb.shape[0]*rgb.shape[1]),3)
    rgbd_reshaped = np.matmul(rgb_reshaped,M2.T).astype(int)
    rgbd = rgbd_reshaped.reshape(rgb.shape)
    rgbd_final = np.clip(rgbd,0,255)
    return rgbd_final

#File Path for train dataset
TRAIN_DIR = "train"
LMS_FINAL_TRAIN_DIR = "trainlms"
RGB_FINAL_TRAIN_DIR = "trainrgb"
LMS_CAT_TRAIN = "trainlms\\Train\\Cat"
RGB_CAT_TRAIN = "trainrgb\\Train\\Cat"
LMS_DOG_TRAIN = "trainlms\\Train\\Dog"
RGB_DOG_TRAIN = "trainrgb\\Train\\Dog"

#File Path for test dataset
LMS_CAT_TEST = "trainlms\\Test\\Cat"
RGB_CAT_TEST = "trainrgb\\Test\\Cat"
LMS_DOG_TEST = "trainlms\\Test\\Dog"
RGB_DOG_TEST = "trainrgb\\Test\\Dog"

#Img Name as list
train_img = os.listdir(TRAIN_DIR)


for img_name in train_img:
    img_class,new_img_name = img_name.split(".")[0:2]
    if int(new_img_name)<11500:
        image = Image.open(TRAIN_DIR + '/' +img_name)
        image_matrix = np.asarray(image)
        rgbd_matrix = rgb2rgbd(image_matrix)
        rgbd_image = Image.fromarray(rgbd_matrix.astype(np.uint8))
    else:
        continue
    
    if int(new_img_name)<10000:
        #Train Dataset    
        if img_class== "cat":
            rgbd_image.save(LMS_CAT_TRAIN + '/' + new_img_name + ".jpg")
            shutil.copy(TRAIN_DIR + '/' +img_name,RGB_CAT_TRAIN + '/' + new_img_name + ".jpg")
        elif img_class== "dog":
            rgbd_image.save(LMS_DOG_TRAIN + '/' + new_img_name + ".jpg")
            shutil.copy(TRAIN_DIR + '/' +img_name,RGB_DOG_TRAIN + '/' + new_img_name + ".jpg")
        else:
            print("Something Went Wrong")
    elif int(new_img_name)<11500:
        #Test Dataset
        if img_class== "cat":
            rgbd_image.save(LMS_CAT_TEST + '/' + new_img_name + ".jpg")
            shutil.copy(TRAIN_DIR + '/' +img_name,RGB_CAT_TEST + '/' + new_img_name + ".jpg")
        elif img_class== "dog":
            rgbd_image.save(LMS_DOG_TEST + '/' + new_img_name + ".jpg")
            shutil.copy(TRAIN_DIR + '/' +img_name,RGB_DOG_TEST + '/' + new_img_name + ".jpg")
        else:
            print("Something Went Wrong")
    else:
        continue
