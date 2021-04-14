# to preprocess image
from keras.preprocessing.image import load_img
from keras.applications.xception import preprocess_input

# models
from keras.applications.xception import Xception
from keras.models import Model
model = Xception()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# for everything else
import os
from colorama import Fore, Back , Style
import numpy as np
from scipy.spatial import distance
import yaml
import shutil
import datetime
from PIL import Image
from utils_obj.obj_tracker import read_classes

# to plot boxes

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes

COLOR = {
    'blue': '\033[94m',
    'default': '\033[99m',
    'grey': '\033[90m',
    'yellow': '\033[93m',
    'black': '\033[90m',
    'cyan': '\033[96m',
    'green': '\033[92m',
    'magenta': '\033[95m',
    'white': '\033[97m',
    'red': '\033[91m',
    'end': '\033[0m ',
    'bold': '\033[1m',
}

def extract_features(file, model = model):
    # load the image as a 299*299 array
    img = load_img(file, target_size=(299,299))
    # img = file.resize(299,299)
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 299,299, 3)
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
    # b =  b.reshape(-1, 4096)
    b =  b.reshape(-1, 2048)

    return (b)


def vectorize(feat_vec_path, im_path):
    """
    method to vectorize multiple images
    :param path is the path where to find the training images
    img_format
    image is added only if img sim is not greater than a th.
    """
    # this list holds all the image filename
    pic = [os.path.join(im_path, i) for i in os.listdir(im_path) if i.split('.')[-1].lower() in img_formats]
    data = []
    c = 0
    not_cop = 0
    where = []
    ind = 0
    for i in pic:
        if i == pic[0]:
            feat = extract_features(i, model)
            data.append(feat)
        else:
            # try to extract the features and update the list
            try:
                feat = extract_features(i, model)
                sim = check_sim(feat, data)
                if not sim:
                    data.append(feat)
                else:
                    not_cop+=1
                    where.append(ind)
            # if something fails, save the extracted features as a pickle file (optional)
            except Exception as e:
                print(e)
                c+=1
        ind +=1
    del_im(where, im_path)

    # save features in txt file
    feat = np.asarray(data)
    mat = np.matrix(feat)

    vec_path = feat_vec_path #os.path.join(path,'feat_vectors.txt' )
    with open(vec_path,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')
        f.close()
    return(not_cop)

def check_sim(vec, data, sim_th=0.95, obj=0):
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

def remove_sim(feat_vec_path, im_path, create_feat=False,  sim_th=0.95):
    if create_feat: # if data are not vectorized already
        not_copied = vectorize(feat_vec_path, im_path)
        if not_copied!=0:
            print('Done. %d images were not vectorized and have been removed from the dataset because very similar.' %not_copied)
        else:
            print('Done. All the images were vectorized.')

    else:
        data = get_feat_vec_data(feat_vec_path)
        print('Checking similarities among %d images...' %len(data))

        where = []
        for i in range(len(data)):
            cur = data[i]
            dc = np.delete(data,i,0)
            res = check_sim(cur, dc, sim_th)
            if res:
                data[i] = np.full(shape = cur.shape, fill_value=-1)
                where.append(i)

        if where:
            print('%d images are similar. \nRemoving...' %(len(where)))
            del_im(where, im_path)
            res = np.delete(data, np.where(np.all(k == -1 for k in data)), axis=0)
            mat = np.matrix(res)

            with open(feat_vec_path, 'wb') as f:
                for line in mat:
                    np.savetxt(f, line, fmt='%.2f')
            f.close()
            print('Done. Removed \033[91m%d\033[0m  similar images' %(len(where)))
        else:
            print('Done. No images where removed.')

def del_im(where, im_path):
    im = [i for i in os.listdir(im_path) if i.split('.')[-1].lower() in img_formats]
    c = os.path.join(im_path, 'removed')
    if not os.path.exists(c):
        os.makedirs(c)

    for i in where:
        ir = im[i]
        shutil.copyfile(os.path.join(im_path, ir), os.path.join(c, ir))
        os.remove(os.path.join(im_path, ir))

def update_feat_matrix(data, feat_vec_path):
    try:
        mat = np.matrix(data)
        with open(feat_vec_path, 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')
            f.close()
        return ('New features vector data updated')
    except Exception as e:
        print('Error occurred when updating the new feature dataset: ', e)

def array2image(im_array):

    img = Image.fromarray(im_array, 'RGB')
    b, g, r = img.split()
    im = Image.merge("RGB", (r, g, b))
    path = 'my_original.png'
    im.save(path)
    return(path)

class Sim(object):
    def __init__(self, yaml_file):

        with open(yaml_file, 'r') as f:
            self.d = yaml.full_load(f)
        f.close()
        self.save_new_img_path = self.d['temp_tr_data_path']
        vectors = self.d['feat_vec_path']
        self.data = get_feat_vec_data(vectors).copy()
        self.added = {} # image path: vector, so it's easier at teh end copy only the images actually labeled
        self.last_simil = 0
        self.classes, _ = read_classes(yaml_file)
        self.skip = 1 # frames to skip similarity check.

    def info(self, fps, save_dir):
        self.fps = fps
        self.temp_pic = str(save_dir)+'/suggested_annot/'  # './temp_label/'
        if not os.path.exists(self.temp_pic):
            os.makedirs(self.temp_pic)

    def new_im(self, im, frame):
        """
        vectorize image and check similarity
        if similar, do nothing, else put vector in matrix of vectors and in dict of im:vec, and copy image in temp_pic
        """
        res = 'sim'
        self.current_frame = frame
        if self.current_frame - self.last_simil >= self.skip*self.fps:
            new_vec=''
            try:
                self.path_im = array2image(im)
                new_vec = extract_features(self.path_im, model)
                # new_vec = extract_features(im, model)
            except Exception as e:
                print('\033[91mSomething went wrong in vectorization\033[0 : %s' %e)


            if len(new_vec):
                similar = check_sim(new_vec, self.data)
                if similar:
                    self.last_simil = frame
                    self.skip += 3
                    # if images are similar, increase the skipping, because is very probable that next frames are similar
                    # and I don't want to see it (I want to save resources)
                else:
                    # if images are not similar, I want to save the image to a temp folder
                    self.data = np.concatenate((self.data, new_vec))
                    name = datetime.datetime.utcnow().strftime("%Y-%m-%d-%Hh-%Mm-%Ss-%fmics")+'_original.jpg'
                    self.new_temp_path = os.path.join(self.temp_pic,name)
                    self.added[name] = new_vec
                    # self.path_im = array2image(im)
                    shutil.copy(self.path_im, self.new_temp_path)
                    #self.skip = 1
                    # print('\033[91mnot sim\033[0m')
                    res = 'not_sim'
                    # if images are not similar, reset skip step to 1
        return(res)

    def save_detection(self, lines):
        # we need to save this detection in case the inspector wants to label them.

        with open(self.new_temp_path[:-4] + '.txt', 'a') as f:
            for line in lines:
                f.write(' '.join(str(i) for i in line))
                f.write('\n')

        self.new_temp_path = ''

    def end(self):
        if self.added:
            print(Back.GREEN + 'New images available for labeling are temp saved in '+Fore.RED+ self.temp_pic + Style.RESET_ALL)

            i = False
            while not i:
                value = input("Do you want to see new images available for labeling? \n \033[1m[Y] or [N] \033[0m \n " )
                if value.lower() == 'y':
                    self.render_images()
                if value.lower() not in ['y', 'n']:
                    print('\033[91mError!\033[0m Please type \033[1m[Y] or [N] \033[0m \n.')
                else:
                    i = True



        else:
            print('\033[92mThis video contained images similar to what we already have in the server. Thank you!\033[0m')

    def render_images(self):
        ann = []
        # t = self.temp_pic
        #
        # im = [os.path.join(t,i) for i in os.listdir(t) if i.split('.')[-1].lower() in img_formats]
        # im = list(filter(None, im))
        #
        # for i in im:
        #     image = Image.open(i)
        #     image.show()

        # TODO:
        # call the labeling here. Specify in which folder you want to save images and labels.
        # this is read in the yaml file. in the init it is stored as: self.save_new_img_path = self.d['temp_tr_data_path']
        # append the vector of images in the feat_vector txt file
        # the vector of the labeled images needs to be added in the feature vector of the training!
        import subprocess

        img_dir = os.path.abspath(self.temp_pic)
        l = r"C:\Users\Giulia Ciaramella\labeling_tool\labelImg-master\labelImg_after_detection.py"
        label_tool_path = os.path.abspath(l)
        clfile = os.path.join(img_dir, "classes.txt")
        predefined_class_file = os.path.abspath(clfile)

        # create txt in the same folder

        with open(predefined_class_file, 'w') as f:
            for c in self.classes.keys():
                f.write(c + '\n')
            f.close()

        label_tool_path = '"' + label_tool_path + '"'
        img_dir = '"' + img_dir + '"'
        predefined_class_file = '"' + predefined_class_file + '"'
        import subprocess
        cmd = 'python ' + label_tool_path + \
              ' --image_dir ' + img_dir + \
              ' --save_dir ' + img_dir + \
              ' --predefined_classes_file '+ predefined_class_file
        s = subprocess.call(cmd, shell=True)


if __name__=='__main__':
    im_path = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\images_new_resized - Copia - Copia'
    v_path = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\images_new_resized - Copia - Copia\feat_vectors_Xception.txt'
    # remove_sim(v_path, im_path, sim_th=0.93)




