from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
import os
import matplotlib.pyplot as plt
# models
from keras.applications.vgg16 import VGG16
from keras.models import Model
from scipy.stats import iqr
import pandas as pd
import pickle
import numpy as np
import warnings
from extract_centroids import get_feat_vec_data
from scipy.spatial import distance
from keras.preprocessing.image import img_to_array
import cv2

warnings.filterwarnings('ignore')

# path = r"C:\Users\Giulia Ciaramella\Documents\archive\flower_images\flower_images"
path = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\images_new_resized'
# change the working directory to the path where the images are located
os.chdir(path)
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def extract_features(file, model = model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


def vectorize(path = r"C:\Users\Giulia Ciaramella\Documents\archive\flower_images\flower_images", img_format = 'png'):

    # this list holds all the image filename
    pic = []
    times = []
    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
        # loops through each file in the directory
        for file in files:
            if file.name.endswith(img_format):
                # adds only the image files to the flowers list
                pic.append(file.name)
    data = []
    c = 0
    for i in pic:
        # try to extract the features and update the list
        try:
            feat = extract_features(i, model)
            data.append(feat)
        # if something fails, save the extracted features as a pickle file (optional)
        except:
            c+=1

    # save features in txt file
    feat = np.asarray(data)
    mat = np.matrix(feat)

    vec_path = os.path.join(path,'feat_vectors.txt' )
    with open(vec_path,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')
        f.close()


# vectorize(path = path, img_format='jpg')
feat = get_feat_vec_data(path = path)
# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1, 4096)
print(len(feat))

stats = False
if stats:
    centroid = np.mean(feat, axis=0)
    feat = sorted(feat)
    m = np.median(feat, axis = 0)
    q1 = np.percentile(feat, 25, axis = 0  )
    q3 = np.percentile(feat, 75, axis = 0 )
    my_iqr = q3-q1
    # iqr_scipy = iqr(feat, axis=0)
    up_lim = q3+my_iqr*1.5
    low_lim = q1-my_iqr*1.5
    range_dist = distance.cosine(up_lim, low_lim)

    # count outliers in the dataset
    def count_outliers(feat):
        o = 0
        for x in feat:
            d1 = distance.cosine(x, low_lim)
            d2 = distance.cosine(x, up_lim)

            if d1+d2 <= range_dist:
                pass
            else:
                o+=1
        print(' outlier in the dataset is', o)


    def is_out(x):
        d1 = distance.cosine(x, low_lim)
        d2 = distance.cosine(x, up_lim)
        if d1 + d2 <= range_dist:
            return False
        else:
            return True
    count_outliers(feat)
    med = is_out(centroid)
    print(med)

    new_image = r'C:\Users\Giulia Ciaramella\Documents\archive\flower_images\flower_images\0006.png'
    im_feat = extract_features(new_image)
    print(is_out(im_feat))

centroid = np.mean(feat, axis = 0)
norms = {distance.cosine(centroid, k):k for k in feat}
max_dist = max(list(norms.keys()))
max_vec = norms[max_dist]


new_image = r'C:\Users\Giulia Ciaramella\Documents\archive\flower_images\flower_images\0006.png'
# new_image = r'C:\Users\Giulia Ciaramella\Desktop\v3d\old\coco\bus.jpg'
# new_image = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\images_new_resized\vlcsnap-2021-02-11-10h40m13s979.jpg'
im_feat = extract_features(new_image)
new_dist =distance.cosine(im_feat,centroid)
if new_dist>max_dist:
    print(new_dist)
    print('Outlier')

    feat2 = np.concatenate((feat, im_feat))
    # recompute centroid and max distance
    centroid2 = np.mean(feat2, axis = 0)
    norms2 = {distance.cosine(centroid2, k): k for k in feat2}
    max_dist2 = max(list(norms2.keys()))
    max_vec2 = norms2[max_dist2]

    new = r'C:\Users\Giulia Ciaramella\Desktop\v3d\old\coco\bus.jpg'
    im = extract_features(new)
    new_dis2 = distance.cosine(im, centroid2)
    if new_dis2 > max_dist2:
        print(new_dis2)
        print('Outlier')

#
#
    plt.scatter(np.arange(len(feat2)), norms2.keys(), c='blue', label='training data')
    plt.scatter(1000, new_dis2, c='red', label='bus')
    plt.xlabel('vectors'), plt.ylabel('cosine distance with centroid')
    plt.scatter(0,0, marker='s', color='orange', label='centroid')
    plt.legend()
    plt.ylim(-0.1,0.85)
    plt.title('asset + flower + bus')
    plt.show()