
import numpy as np
import os
import shutil
from scipy.spatial import distance
feat_path = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\images_new_resized\feat_vectors.txt'
im_path = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\images_new_resized'
def get_feat_vec_data(path):
    if not path.endswith('txt'):
        path = os.path.join(path, 'feat_vectors.txt')

    with open(path,'r') as f:
        raw = []
        for line in f:
            r = line.split(' ')
            raw.append([float(i) for i in r])
        f.close()

    b = np.asarray(raw)
    b =  b.reshape(-1, 2808)
    return (b)

def check_sim(vec, data, sim_th=0.98, obj=0):
    """
    evaluate the similarity between 1 vector and all the other vectors in data.
    Returns True is the new image is very similar to image already present (sim>threshold)
    obj indicates the metric: how many similar images can exist. Default = 0
    """
    sim = [1-distance.cosine(vec, k) for k in data if not np.all((k==-1))]
    max_sim = max(sim)
    # print(max_sim)
    if obj !=0 :
        s = list(filter(lambda x: x>sim_th, sim))
        if len(s) >= obj:
            return True
        else:
            return False
    else:
        if max_sim>=sim_th:
            # print(max_sim)
            return True # new vector is similar to someone that we have and should not be added
        else:
            return False # new vec is different to anyother images and should be added

def del_im(where, im_path):
    im = [i for i in os.listdir(im_path) if i.split('.')[-1].lower() in ['jpg', 'png']]
    c = os.path.join(im_path, 'removed')
    if not os.path.exists(c):
        os.makedirs(c)

    for i in where:
        ir = im[i]
        shutil.copyfile(os.path.join(im_path, ir), os.path.join(c, ir))
        os.remove(os.path.join(im_path, ir))

vecs = get_feat_vec_data(feat_path)

where = []
for i in range(len(vecs)):
    cur = vecs[i]
    dc = np.delete(vecs,i,0)
    res = check_sim(cur, dc)
    if res:
        vecs[i] = np.full(shape = cur.shape, fill_value=-1)
        where.append(i)

print(len(where))
print(where)
if where:
    print('%d images are similar. \nRemoving...' % (len(where)))
    del_im(where, im_path)
    res = np.delete(vecs, np.where(np.all(k == -1 for k in vecs)), axis=0)
    mat = np.matrix(res)

    print('Done. Removed \033[91m%d\033[0m  similar images' % (len(where)))
else:
    print('Done. No images where removed.')

# with open(fv, 'r') as f:
#     raw = []
#     for line in f:
#         r = line.split(' ')
#         raw.append([float(i) for i in r])
#     f.close()
#
# b = np.asarray(raw)
# b = b.reshape(-1, 4096)
#
# sim = 1 - distance.cosine(b[2], b[1])
#
# print(sim)

# from PIL import Image
#
# pic_or = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\images_new_resized\vlcsnap-2021-02-11-10h35m44s403.jpg'
# # open image
# im_or = Image.open(pic_or)
# im_mod = im_or.resize((416,640))
# im_mod.save('mod.jpg')











