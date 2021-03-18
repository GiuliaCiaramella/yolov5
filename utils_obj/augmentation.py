import os
import glob
from PIL import Image
import albumentations as A
import cv2
from skimage.filters import prewitt_h,prewitt_v
import shutil
import random
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from skimage.feature import hog
from skimage import exposure
from skimage import feature
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')

#path = Path("/here/your/path/") #path to (train/images/) folder

def oneof(image_file,path_to_train):
    liste = (canny,canny,canny,canny,canny,canny,canny,canny,canny,canny,
    hog_our,
    albumone,albumone,albumone,albumone,albumone,albumone,albumone,albumone,albumone,albumone,
    albumtwo,albumtwo,albumtwo,albumtwo,albumtwo,albumtwo,albumtwo,albumtwo,albumtwo,albumtwo,
    prewittk,prewittk,prewittk,prewittk,prewittk,prewittk,prewittk,prewittk,prewittk,prewittk,
    )
     #defininng transformation
    random.choice(liste)(image_file,path_to_train)

def canny(image_file,path_to_train):
    im = imread(image_file,as_gray=True )
    edges1 = feature.canny(im)
    plt.imsave(image_file[:-4]+'_c.jpg', edges1, cmap = plt.cm.gray)
    base = os.path.basename(image_file)
    base = base[:-4]+'.txt'
    shutil.copy2(path_to_train+"/labels/"+base,path_to_train+"/labels/"+base[:-4]+'_c.txt')



def prewittk(image_file,path_to_train):
    im = imread(image_file,as_gray=True )                         
    edges_prewitt_horizontal = prewitt_h(im)
    edges_prewitt_vertical = prewitt_v(im)
    imsave(image_file[:-4]+'_p.jpg',edges_prewitt_vertical, cmap='gray')
    base = os.path.basename(image_file)
    base = base[:-4]+'.txt'
    shutil.copy2(path_to_train+"/labels/"+base,path_to_train+"/labels/"+base[:-4]+'_p.txt')


def hog_our(image_file,path_to_train):
    image =Image.open(image_file)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(3, 3),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    imsave(image_file[:-4]+'_h.jpg',hog_image_rescaled)
    base = os.path.basename(image_file)
    base = base[:-4]+'.txt'
    shutil.copy2(path_to_train+"/labels/"+base,path_to_train+"/labels/"+base[:-4]+'_h.txt')  


def albumone(image_file,path_to_train):
    transform = A.Compose([
      A.RandomBrightnessContrast(p=0.4),
     A.OneOf([
              A.IAAAdditiveGaussianNoise(),
              A.GaussNoise(),
          ], p=0.3),
          A.OneOf([
              A.MotionBlur(p=.5),
              A.MedianBlur(blur_limit=3, p=0.6),         #mrandom
              A.Blur(blur_limit=3, p=0.6),
          ], p=0.2),
          A.HueSaturationValue(p=0.7),
          A.RGBShift(p=0.6),
          A.InvertImg(p=0.4),
    ])
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    cv2.imwrite(image_file[:-4]+'_aone.jpg', transformed_image)
    base = os.path.basename(image_file)
    base = base[:-4]+'.txt'
    shutil.copy2(path_to_train+"/labels/"+base,path_to_train+"/labels/"+base[:-4]+'_aone.txt')


def albumtwo(image_file,path_to_train):
    transform2 = A.Compose([
        A.RandomBrightnessContrast(p=0.6),
        A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.4),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.HueSaturationValue(),
            A.RGBShift(),
    ])
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed2 = transform2(image=image)
    transformed2_image = transformed2["image"]
    cv2.imwrite(image_file[:-4]+'_atwo.jpg', transformed2_image)
    base = os.path.basename(image_file)
    base = base[:-4]+'.txt'
    shutil.copy2(path_to_train+"/labels/"+base,path_to_train+"/labels/"+base[:-4]+'_atwo.txt')

def augment(path_to_train):   #path must be given to images!!!!!
    for image_file in tqdm(glob.iglob(path_to_train+'/images/*.jpg')):
        oneof(image_file,path_to_train)
    print("\nImages and labels are converted!\n")



#augment("/home/kutay/Desktop/dataset/train/")