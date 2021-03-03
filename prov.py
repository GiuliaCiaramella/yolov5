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
import random

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

import cv2
import math

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
# rotation('my_original.png')

# def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
def random_perspective(img, targets=(), degrees=10, translate=0, scale=.01, shear=10, perspective=0.0,
                           border=(0, 0)):

    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            img2 = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))

        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            img2 = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))


    # Visualize
    # import matplotlib.pyplot as plt
    ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    ax[0].imshow(img[:, :, ::-1])  # base
    ax[1].imshow(img2[:, :, ::-1])  # warped
    plt.show()

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        print(xy)
        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        print(i)
        targets = targets[i]
        targets[:, 1:5] = xy[i]


    return img, targets

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def plot_one_box(x, img, objectID = '', color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cx = int((x[2]+ x[0])/2.0) # non c'era
        cy = int((x[1]+ x[3])/2.0) # non c'era
        cv2.circle(img, (cx, cy), 4, color, -1) # to draw the center
        text = "ID {}".format(objectID)
        cv2.putText(img, text, (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # non c'era
        #cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

import os
f = 'vlcsnap-2021-02-11-10h39m32s606'
img = os.path.join(r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\images_new_resized', f+'.jpg')
lab = os.path.join(r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\labels', f+'.txt')

with open(lab, 'r') as f:
    raw = []
    for line in f:
        r = line.split(' ')
        raw.append([float(i) for i in r])
    f.close()
im = skimage.io.imread(img)
# img, labels = random_perspective(img=im, targets=np.asarray(raw))

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


ia.seed(1)

image = im
h, w  = image.shape
bb = []
for b in raw:
    bb.append(BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]))

bbs = BoundingBoxesOnImage(bb, shape=image.shape)

seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
        translate_px={"x": 40, "y": 60},
        scale=(0.5, 0.7)
    ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])

# Augment BBs and images.
image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

# print coordinates before/after augmentation (see below)
# use .x1_int, .y_int, ... to get integer coordinates
for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
    )

# image with BBs before/after augmentation (shown below)
image_before = bbs.draw_on_image(image, size=2)
image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

skimage.io.imshow(image_before)
skimage.io.imshow(image_after)
plt.show()