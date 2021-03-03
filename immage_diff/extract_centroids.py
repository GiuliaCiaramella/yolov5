import numpy as np
import os
p = r'C:\Users\Giulia Ciaramella\Documents\archive\flower_images\flower_images'
pca_def = r'C:\Users\Giulia Ciaramella\Desktop\archive\flower_images\flower_images'
feat_path = r'C:\Users\Giulia Ciaramella\Documents\archive\flower_images\flower_images'

def get_centroids(path = p):
    if not path.endswith('txt'):
        path = os.path.join(path, 'centroids.txt')
    with open(path,'r') as f:
        raw = []
        for line in f:
            r = line.split(' ')
            raw.append([float(i) for i in r])
        f.close()

    b = np.asarray(raw)
    return (b)

def get_pca(path = pca_def):
    if not path.endswith('txt'):
        path = os.path.join(path, 'pca_components.txt')
    with open(path,'r') as f:
        raw = []
        for line in f:
            r = line.split(' ')
            raw.append([float(i) for i in r])
        f.close()

    b = np.asarray(raw)
    return (b)

def get_feat_vec_data(path = feat_path):
    if not path.endswith('txt'):
        path = os.path.join(path, 'feat_vectors.txt')
    with open(path,'r') as f:
        raw = []
        for line in f:
            r = line.split(' ')
            raw.append([float(i) for i in r])
        f.close()

    b = np.asarray(raw)
    return (b)