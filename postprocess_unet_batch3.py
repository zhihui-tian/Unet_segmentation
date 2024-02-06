from PIL import Image
import requests
import matplotlib.pyplot as plt
import cv2
from skimage.filters import sobel
import numpy as np
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage import color
from skimage import morphology
from scipy import stats as st

for i in range(700,900):
    raw_image = Image.open(r"T:\grain_unet-master\Data\test\test_data\fovs120\fov20\predict_512\cropped"+str(i)+'.png')
    raw_image_array=np.array(raw_image)
    raw_image_rgb = cv2.cvtColor(raw_image_array, cv2.COLOR_GRAY2RGB )
    ret1,thresh=cv2.threshold(raw_image_array,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # find threshold iteself, the value is ret1:218
    img_closed = morphology.remove_small_holes(thresh, area_threshold=100)
    thresh=np.where(img_closed,255,0).astype(np.uint8)
    kernel=np.ones((3,3),np.uint8)
    opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)  ### erosion followed by dilation
    sure_bg=cv2.dilate(opening,kernel,iterations=1)  ### make the foreground large to get the sure background
    dist_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)   ### value of each pixel is replaced by its distance to nearest background pixels
    ret2,sure_fg=cv2.threshold(dist_transform,0.25*dist_transform.max(),255,0)
    sure_fg=np.uint8(sure_fg)
    unknown=cv2.subtract(sure_bg,sure_fg)
    ret3,markers=cv2.connectedComponents(sure_fg)
    markers=markers+10   # watershed only works on pixel value are 0, so for sure background and sure foreground, set it not as 0 and only set unknow as 0
    markers[unknown==255]=0
    markers=cv2.watershed(raw_image_rgb,markers)
    np.save("T:/grain_unet-master/Data/test/test_data/fovs120/fov20\predict_512\post_processed/cropper_"+str(i)+".npy",markers)




### create training dataset
k_all=[]
k_old = np.load("T:/grain_unet-master/Data/test/test_data/fovs120/fov20\predict_512\post_processed/cropper_" + str(700) + ".npy")
k_old[(k_old == -1) | (k_old == 10)] = 0
for i in range(np.unique(k_old).shape[0]):
    k_old[(k_old == np.unique(k_old)[i])] = i
k_all.append(k_old)
k_old1=k_old.copy()
for j in range(701,900):
    k_new=np.load("T:/grain_unet-master/Data/test/test_data/fovs120/fov20\predict_512\post_processed/cropper_"+str(j)+".npy")
    k_new[(k_new == -1) | (k_new == 10)] = 0
    k_new_new = np.zeros((512, 512))
    for k in range(1, np.unique(k_new).shape[0]):
        [a, b] = np.where(k_new == np.unique(k_new)[k])  # a行b列
        n = st.mode(k_old1[a, b])[0]
        n = n[0]
        k_new_new[a, b] = n
    k_old1=k_new_new.copy()
    k_all.append(k_new_new)

np.save("./700to900_2.npy",np.array(k_all))