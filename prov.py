# import yaml
#
# p = r'C:\Users\Giulia Ciaramella\PycharmProjects\E2E\general_conf.yaml'
#
# with open(p, 'r') as f:
#     d = yaml.full_load(f)
# f.close()
# print(d['main server'])

from skimage import transform, data
import skimage.io
from skimage.transform import AffineTransform
import matplotlib.pyplot as  plt
import numpy as np
import shutil

def copy_label(orimg_name, trimg_name):

    return

# implement crop
def crop(im):
    pass

# implement shear
def shear(im_path, alpha = 0.3):
    sh = {'p': 0 + alpha, 'n': 0 - alpha}
    format = im_path.split('.')[-1]
    name = im_path[:-4] + 'shear'
    name = name.replace('original', '')

    for r in list(sh.keys()):
        n = name + '_' + r + '.' + format
        im = skimage.io.imread(im_path)
        tform = AffineTransform(shear=sh[r])
        img_warp = transform.warp(im, tform)
        img_warp = (img_warp * 255).astype(np.uint8)

        skimage.io.imshow(img_warp)
        skimage.io.imsave(n, img_warp)

    # modify label



# implement rotation
def rotation(im_path, alpha = 0.1):
    rot = {'p': 0+alpha, 'n':0-alpha}
    format = im_path.split('.')[-1]
    name = im_path[:-4] + 'rot'
    name = name.replace('original', '')

    for r in list(rot.keys()):
        n = name + '_' +r + '.' + format
        im = skimage.io.imread(im_path)
        tform = AffineTransform(shear=rot[r])
        img_warp = transform.warp(im, tform)
        img_warp = (img_warp * 255).astype(np.uint8)

        skimage.io.imshow(img_warp)
        skimage.io.imsave(n, img_warp)

# shear('my_original.png')
rotation('my_original.png')