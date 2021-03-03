# read new pic
# compute NN prediction
# extract feature vector
# compute pca
# calculate distance with centroids
import numpy as np
from scipy.spatial import distance
from extract_centroids import get_pca, get_centroids
import os
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

from sklearn.decomposition import PCA

# 0002.png is in class 0
path = r"C:\Users\Giulia Ciaramella\Documents\archive\flower_images\flower_images"
img = os.path.join(path, '0095.png') # im 0027.png is class 6
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
    return  np.array(features)


# apply pca for dimensionality reduction
feat = extract_features(img)

feat = feat.reshape(1,-1) # 1 because we have only one sample

# delete the features less important

i = 0
# get centroids
centroids = get_centroids()
print(centroids.shape)
# evaluate distance
# dist = np.linalg.norm(point1 - x)
eval_cosine = []
eval_sim = []
eucl = []
for c in centroids:
    eucl.append(np.linalg.norm(c-feat))
    d = distance.cosine(c, feat)
    eval_cosine.append(d)
    eval_sim.append(1-d)


cos_dist = min(eval_cosine)
print('cos dist',eval_cosine.index(cos_dist))

max_simil = max(eval_sim)
print('cos sim', eval_sim.index(max_simil))

min_dist = min(eucl)
print(eucl.index(min_dist))
# the metric used to evaluate the similitudine of vectors is cosin similarity.
# this measure the similarity between cosines of vectors