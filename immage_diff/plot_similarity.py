import numpy as np
import matplotlib.pylab as plt
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
import shutil
import os
import time

from PIL import Image
# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

import warnings
from extract_centroids import get_feat_vec_data
from scipy.spatial import distance


# first vectorize the data in 'training'
path = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\image_divers\data_set_vectorized'
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

def vectorize(path =path, img_format = 'jpg'):

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

def vectorize_one(im =path, model = model):
    # this list holds all the image filename
    c = 0

    try:
        feat = extract_features(im, model)
        return(feat)
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        print('no')
        c+=1
        return ''


already_present = []
def check_sim(vec, data, sim_th=0.95):
    global already_present
    sim = [1-distance.cosine(vec, k) for k in data]
    max_sim = max(sim)
    if max_sim>=sim_th:
        already_present.append(sim.index(max_sim))
        return True # new vector is similar to someone that we have and should not be added
    else:
        return False # new vec is different to anyother images and should be added

# vectorize(path = path, img_format='jpg')
feat = get_feat_vec_data(path = path)
# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1, 4096)

# new entries
path_new_entries = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\image_divers\new_entries'
new_entries = [i for i in os.listdir(path_new_entries) if i.endswith('jpg')]

not_added = []
check_time = []
stats = {'check_time':[], 'data':[], 'vec_time':[], 'total_time':[]}

for new in new_entries:
    t0 = time.time()
    d = len(feat)
    new_vec = vectorize_one(im = os.path.join(path_new_entries, new))
    stats['vec_time'].append(time.time()-t0)
    t1 = time.time()
    if new_vec.shape:
        if check_sim(new_vec,feat): # already similar image exists
            not_added.append(new)
        else:
            feat = np.concatenate((feat, new_vec))
    t2 = time.time()
    stats['check_time'].append(t2-t1)
    stats['data'].append(d)
    stats['total_time'].append(t2-t0)

if not len(not_added):
    print('All the images have been added')
else:
    # elements that returned a similarity >0.95
    not_copied_path = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\image_divers\not copied'
    for filename in os.listdir(not_copied_path):
        os.remove(os.path.join(not_copied_path, filename))

    for i in not_added:
        shutil.copy(os.path.join(path_new_entries, i), not_copied_path)

    # # # original images from similarity
    # original_similar = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\image_divers\original_similar'
    # for filename in os.listdir(original_similar):
    #     os.remove(os.path.join(original_similar, filename))
    #
    # listdir = os.listdir(path) # ho tutte le immagini
    # # already_present ho indici delle immagini
    #
    # listdir.pop(0)
    # # print(len(listdir))# should be 719
    # # print(already_present)
    # for i in already_present: # contains indeces
    #     try:
    #         name = listdir[i]
    #         if not name.endswith('txt'):
    #             shutil.copy(os.path.join(path, os.listdir(path)[i]), original_similar)
    #     except:
    #         name = listdir[i-len(listdir)]
    #         if not name.endswith('txt'):
    #             shutil.copy(os.path.join(path_new_entries, os.listdir(path_new_entries)[i-len(listdir)]), original_similar)
    #

    print(len(not_added))



# moving average
window_size = 6
i = 0
check_moving_averages = []
while i < len(stats['check_time']) - window_size + 1:

    # checking time moving average
    check_this_window = stats['check_time'][i : i + window_size]
    check_window_average = sum(check_this_window) / window_size
    check_moving_averages.append(check_window_average)

    # # checking time moving average
    # check_this_window = stats['check_time'][i: i + window_size]
    # check_window_average = sum(check_this_window) / window_size
    # check_moving_averages.append(check_window_average)

    i += 1

fig = plt.figure()
plt.plot(stats['data'], stats['check_time'], label='check_time', c='red')
plt.plot(stats['data'], stats['vec_time'], label='vec_time', c='orange')
plt.plot(stats['data'], stats['total_time'], label='total_time', c='blue')
plt.xlabel('data len'), plt.ylabel('time')
plt.legend()
plt.show()

plt.plot(stats['data'], stats['check_time'], label='check_time', c='red')
plt.plot(np.arange(stats['data'][0]+window_size, stats['data'][0]+window_size+len(check_moving_averages)),
         check_moving_averages, c='black', label='moving average (wsize=6)')
plt.xlabel('data len'), plt.ylabel('time')
plt.legend()
plt.show()
