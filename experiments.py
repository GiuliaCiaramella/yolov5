import yaml
import numpy as np
from scipy.spatial import distance
import pandas as pd
import warnings
import os

warnings.filterwarnings(action='ignore')
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
start_time = 3 * 60 + 43
end_time = 9 * 60 + 25
ps = r'F:\VivaDrive\v3d\19_03_e2e_presentation\presentation_19_03_version5.mp4'
input_video = ps
output_video = r'F:\VivaDrive\v3d\19_03_e2e_presentation\presentation_19_03_version5_cut.mp4'
ffmpeg_extract_subclip(input_video, start_time, end_time, targetname=output_video)
print('done')

import glob
from keras_preprocessing.image import load_img
from keras.applications.xception import  preprocess_input
from keras.models import Model
from keras.applications import Xception
model = Xception()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)


fvs = []
p = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\images_new_resized - Copia'
v = os.path.join(p, 'feat_vectors_fromXception.txt')
images = [i for i in os.listdir(p) if i.endswith('jpg')]
i = 0
def extract_feat(file):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(299, 299))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 299, 299, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

def get_feat_vec_data(path):

    """
    read the txt file containing the vectors of images and return the text as a matrix of arrays
    """
    if not path.endswith('txt'):
        path = os.path.join(path, 'feat_vectors.txt')
    with open(path,'r') as f:
        raw = []
        for line in f:
            r = line.split(' ')
            raw.append([float(i) for i in r])
        f.close()

    b = np.asarray(raw)
    b =  b.reshape(-1, 4096)
    print(b.shape)
    return (b)


vectorize = True
if vectorize:
    for im in glob.glob(p+'/*.jpg'):
        i+=1
        print(i)
        try:
            feat = extract_feat(im) # shape = (1,2048)
            fvs.append(feat)
        except Exception as e:
            print(e)

    try:
        vec_path = os.path.join(p,'./feat_vectors_Xception.txt')  # os.path.join(path,'feat_vectors.txt' )
        # fvs_array = fvs.detach().cpu().numpy()
        fvs_arr = np.asarray(fvs)
        mat = np.matrix(fvs_arr)
        print(mat.size)
        with open(vec_path, 'wb') as f:
            for line in mat:
                np.savetxt(f, line)
            f.close()

    except Exception as e:
        print(e)

