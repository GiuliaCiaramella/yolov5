from sklearn.cluster import KMeans
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import cv2
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.image import load_img

path = r"C:\Users\Giulia Ciaramella\Desktop\archive\flower_images\flower_images"
# change the working directory to the path where the images are located
os.chdir(path)

# this list holds all the image filename
flowers = []
with os.scandir(path) as files:
    # loops through each file in the directory
    for file in files:
        if file.name.endswith('.png'):
            # adds only the image files to the flowers list
            flowers.append(file.name)



def apply_k(file, k=3):

    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32,32))

    # Next, converts the MxNx3 image into a Kx3 matrix where K=MxN and each row is now a vector in the 3-D space of RGB.
    vectorized = img.reshape((-1, 3))
    vectorized = np.float32(vectorized)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # center = np.uint8(center)
    # res = center[label.flatten()]
    # result_image = res.reshape((img.shape))
    # figure_size = 15
    # plt.figure(figsize=(figure_size, figure_size))
    # plt.subplot(1, 2, 1), plt.imshow(img)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(result_image)
    # plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
    # plt.show()

    return label

data = {}
p = r"C:\Users\Giulia Ciaramella\PycharmProjects\yolov5\immage_diff"
for flower in flowers:
    # try to extract the features and update the dictionary
    try:
        feat = apply_k(flower)
        data[flower] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p, 'wb') as file:
            pickle.dump(data, file)

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))
l = len(feat[0])
# reshape so that there are 210 samples of 32*32 vectors
feat = feat.reshape(-1, l)

# get the unique labels (from the flower_labels.csv)
df = pd.read_csv('flower_labels.csv')
label = df['label'].tolist()
unique_labels = list(set(label))

x = feat
# cluster feature vectors
kmeans = KMeans(n_clusters=len(unique_labels), n_jobs=-1, random_state=22)
# kmeans = KMeans(n_clusters=5, n_jobs=-1, random_state=22)

kmeans.fit(x)
center = kmeans.cluster_centers_
mat = np.matrix(center)

with open('centroid.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.2f')
    f.close()
i = 0


# holds the cluster id and the images { id: [images] }
groups = {}

for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)


# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize=(25, 25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

for i in range(len(unique_labels)):
    view_cluster(i)

# this is just incase you want to see which value for k might be the best
sse = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22, n_jobs=-1)
    km.fit(x)

    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');
plt.show()

i=0
