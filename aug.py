from skimage import transform, data
import skimage.io
from skimage.transform import AffineTransform
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import imageio


def bbs2xywhn(bbs):
    global h, w

    box = [bbs.x1, bbs.y1, bbs.x2, bbs.y2]
    b = box.copy()

    b[0] = (box[0] + box[2]) / 2/w  # x center
    b[1] = (box[1] + box[3]) / 2/h  # y center
    b[2] = (box[2] - box[0])/w  # width
    b[3] = (box[3] - box[1])/h # height

    return b

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y
    return y

def aug_type(tform):
    seq = ''
    if tform == 'rotate':
        seq = iaa.Sequential([
            iaa.Affine(
                rotate=15
            )
        ])
    elif tform == 'shear':
        seq = iaa.Sequential([
            iaa.Affine(
                shear=15
            )
        ])
    return(seq)

def pipeline(img, lab, tr=['rotate', 'shear']):
    with open(lab, 'r') as f:
        raw = []
        complete_row = []
        for line in f:
            r = line.split(' ')
            no_class = r[1:] # remove class
            raw.append([float(i) for i in no_class])
            complete_row.append(r[0])
        f.close()
    image = skimage.io.imread(img)
    global h, w, _
    h, w, _ = image.shape

    bb = []
    for b in raw:
        # convert from xywhn to xyxy not normalized
        b = xywhn2xyxy(b, w, h )
        bb.append(BoundingBox(x1=b[0], y1=b[3], x2=b[2], y2=b[1]))

    bbs = BoundingBoxesOnImage(bb, shape=image.shape)

    for r in tr:
        print(r)
        seq= aug_type(r)
        new_lab = lab.replace('original', r)
        # Augment BBs and images.
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

        image_before = bbs.draw_on_image(image, size=2)
        image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 246])
        # for i in bbs_aug.bounding_boxes:
        #     print("(%.4f, %.4f, %.4f, %.4f)" % (
        #         i.x1, i.y1, i.x2, i.y2)
        #     )

        ia.imshow(image_before)
        ia.imshow(image_after)
        imageio.imwrite(img.replace('original', r),image_aug)

        # save the new txt
        with open(new_lab, 'w') as f:
            for i in range(len(bbs_aug)):
                # from bbs to xywhn
                xywhn = bbs2xywhn(bbs_aug[i])
                line = str(complete_row[i]) + ' '+ ' '.join([str(elem) for elem in xywhn])
                f.write(line)

        new_lab=''
        f.close()



img = 'prova_original.jpg'
lab = 'prova_original.txt'
# img = 'prova_shear.jpg'
# lab = 'prova_shear.txt'

pipeline(img, lab)