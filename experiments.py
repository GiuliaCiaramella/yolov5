import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import skimage
import skimage.feature
import skimage.viewer
import skimage.io
import scipy.ndimage as nd
from skimage.filters import sobel, laplace
import os
import shutil
import yaml

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

file_name = r'../data/images/zidane.jpg'
# img = cv2.imread(r'data/images/zidane.jpg')

cluster = False
if cluster:
    img = cv2.imread(r"C:\Users\Giulia Ciaramella\Desktop\archive\flower_images\flower_images\0006.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Next, converts the MxNx3 image into a Kx3 matrix where K=MxN and each row is now a vector in the 3-D space of RGB.
    vectorized = img.reshape((-1,3))
    # We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV.
    vectorized = np.float32(vectorized) # matrice (HxW)x3 dove H sono i pizel in altezza, W i px in larghezza e 3 sono RGB

    #Define criteria, number of clusters(K) and apply k-means()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    # Now convert back into uint8.
    center = np.uint8(center)

    A = vectorized[label.flatten() == 0]
    B = vectorized[label.flatten() == 1]
    # Plot the data
    plt.scatter(A[:, 0], A[:, 1])
    plt.scatter(B[:, 0], B[:, 1], c='r')
    plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    plt.xlabel('White'), plt.ylabel('Black')
    plt.show()

    #Next, we have to access the labels to regenerate the clustered image
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(result_image )
    plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
    plt.show()
    print('hey')



# plot vectorized: - solo per dimostrare cosa contiene l'array vectorized
# è lento da stampare se ci sono molti punti. in realtà l'algoritmo è motlo più veloce
plot = False
if plot:
    r, g, b = np.asarray([item for item in vectorized[:,0]]), np.asarray([item for item in vectorized[:,1]]), np.asarray([item for item in vectorized[:,2]])
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()#plotting
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('r')
    ax.set_ylabel('g')
    ax.set_zlabel('b')
    ax.scatter(r, g, b)
    plt.show()

def can(file, cv_canny = False, scipy_canny = False, scikit_canny = False):
    if cv_canny:
        #figure_size = 15
        # source = cv2.imread(r'data/images/zidane.jpg',3) # read image
        source = cv2.imread(file,3) # read image
        b, g, r = cv2.split(source)  # get b, g, r
        rgb_img1 = cv2.merge([r, g, b])  # switch it to r, g, b
        img = cv2.GaussianBlur(rgb_img1, (3, 3), 0) # remove noise
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert in grayscale

        edges = cv2.Canny(src_gray,100,300, False)
        return(edges)

        # vectorized = edges.reshape((-1, 1))
        # # We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV.
        # vectorized = np.float32(vectorized) # vector containing for each pixel 255 if white (edge) and 0 if black
        # i = 0

        ## see results of edge extraction
        #cv2.imwrite('prova_c.png', edges)

        # plt.figure(figsize=(figure_size,figure_size))
        # plt.subplot(1,2,1),plt.imshow(rgb_img1)
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(1,2,2),plt.imshow(edges,cmap = 'gray')
        # plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
        # plt.show()
        #print('done in ', stop-start)

    if scikit_canny:
        s = time.time()
        # load and display original image as grayscale
        image = skimage.io.imread(fname=file_name, as_gray=True)
        #view = skimage.viewer.ImageViewer(image)
        #view.show()
        canny_edges = skimage.feature.canny(
            image=image,
            sigma=1.5, # the higher, the smoother, regulates the noise filter (gaussian) in the image
            low_threshold=0.100,
            high_threshold=0.300,
        )
        p = time.time()
        #viewer = skimage.viewer.ImageViewer(canny_edges)
        #viewer.show()
        print('scikit canny done in: ', p-s)

def sob(file, cv_sobel = False, scipy_sobel = False, scikit_sobel = False):
    if cv_sobel:
        img = cv2.imread(file)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dst = cv2.Sobel(src_gray, cv2.CV_8U, 1, 0, ksize=3) # with cv2.cv_8u we loose edges in pixels from withe to black transition
        return(dst)


    if scipy_sobel:
        plt.gray()  # show the filtered result in grayscale
        start = time.time()
        img = cv2.imread(file_name)
        # first we need to convert in gray scale the image
        red = img[:,:,0]
        green = img[:,:,1]
        blue = img[:,:,2]
        grey = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)

        result = nd.sobel(grey, 1)
        stop = time.time()
        plt.imshow(result)
        plt.show()
        print('done in ', stop-start)

    if scikit_sobel:
        start = time.time()
        img = cv2.imread(file_name)
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
        grey = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
        dst = sobel(grey)
        stop = time.time()
        print('done in ', stop-start)

        plt.imshow(dst, cmap='gray')
        plt.show()
def lap(cv_laplacian = False, scipy_laplacian = False, scikit_laplacian = False):
    if cv_laplacian:
        ddepth = cv2.CV_8U # _16S
        kernel_size = 3
        window_name = "Laplace"

        start = time.time()
        img = cv2.imread(file_name)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
        abs_dst = cv2.convertScaleAbs(dst)

        # dst = cv2.Laplacian(grey, cv2.CV_8U)
        # dst = cv2.Laplacian(grey, cv2.CV_64F)

        # dst =  cv2.Laplacian(grey,cv2.CV_64F)
        stop = time.time()
        print('done in ', stop - start)
        plt.imshow(abs_dst, cmap='gray')
        plt.show()

    if scipy_laplacian:
        plt.gray()  # show the filtered result in grayscale
        start = time.time()
        img = cv2.imread(file_name)
        # first we need to convert in gray scale the image
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
        grey = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)

        result = nd.laplace(grey)
        stop = time.time()
        print('done in ', stop - start)

        plt.imshow(result)
        plt.show()

    if scikit_laplacian:
        start = time.time()
        img = cv2.imread(file_name)
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
        grey = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)

        dst = laplace(grey,ksize=3)
        stop = time.time()
        print('done in ', stop-start)

        plt.imshow(dst, cmap='gray')
        plt.show()



# # lap(cv_laplacian=True)
# can(cv_canny=True)
# sob(cv_sobel=True)

transform = False
if transform:
    # take 1 image per time and transform it
    p = r'C:\Users\Giulia Ciaramella\Desktop\modified_st_raw\images_new_resized'
    l = r'C:\Users\Giulia Ciaramella\Desktop\modified_st_raw\labels'
    sav_path_img = r'C:\Users\Giulia Ciaramella\Desktop\new_aug_edges\img'
    sav_path_lab = r'C:\Users\Giulia Ciaramella\Desktop\new_aug_edges\lab'


    pic_name = [f for f in os.listdir(p)]
    label_name = [f for f in os.listdir(l)]

    for a in label_name:
        i = a[:-3]+'jpg'
        canny_edge = can(file = os.path.join(p,i), cv_canny=True)
        cv2.imwrite(os.path.join(sav_path_img,i[:-4]+'_canny.jpg'), canny_edge)
        # copy also label in the directory
        shutil.copyfile(os.path.join(l,a), os.path.join(sav_path_lab,a[:-4]+'_canny.txt' ))

        sob_edge = sob(file = os.path.join(p,i), cv_sobel=True)
        cv2.imwrite(os.path.join(sav_path_img,i[:-4]+ '_sobel.jpg'), sob_edge)
        shutil.copyfile(os.path.join(l,a), os.path.join(sav_path_lab,a[:-4]+'_sobel.txt' ))

general_yaml_file = r'C:\Users\Giulia Ciaramella\PycharmProjects\E2E\general_conf.yaml'

with open(general_yaml_file) as file:
    assets = yaml.full_load(file)
file.close()

# print(" ---- ".join(map(str,assets.keys())))
# value = input("Please choose an asset. You can choose among: \n \033[1m%r" %"     ".join(map(str,assets.keys())))
# print(f'you choose: ', value)
i = False
# while not i:
#     value = input("Please choose an asset. You can choose among: \n \033[1m%r\033[0m \n " % "   ".join(map(str, assets.keys())))
#     if value not in list(assets.keys()):
#         print('\033[91mError!\033[0m The asset you chose is not in the list.')
#     else:
#         i = True
# yaml_file = assets[value]
# with open(yaml_file) as file:
#     new = yaml.full_load(file)
# file.close()
# print(new['weight_file'])

# import datetime
# print(datetime.datetime.utcnow().strftime("%Y-%m-%d-%Hh-%Mm-%Ss-%fmics") )
# im = r'C:\Users\Giulia Ciaramella\Desktop\v3d\edge_data\modified_st_raw\images_new_resized\vlcsnap-2021-02-11-10h35m44s403.jpg'
# shutil.copy(im, os.path.join('./', datetime.datetime.utcnow().strftime("%Y-%m-%d-%Hh-%Mm-%Ss-%fmics")+'_original.jpg'))

# pip install --trusted-host pypi.python.org moviepy
# pip install imageio-ffmpeg

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
start_time = 5*60+10
end_time= 6*60+26
input_video = r'F:\VivaDrive\v3d\BASF-F866-C200\F866-C200\123_0038.MOV'
output_video = r'F:\VivaDrive\v3d\BASF-F866-C200\F866-C200\123_0038_cut.MOV'
# ffmpeg_extract_subclip(input_video, start_time, end_time, targetname=output_video)




import glob
# import cv2
#
# img_dir = os.path.abspath('../runs/detect/exp4/suggested_annot')
# label_tool_path = os.path.abspath('../utils/labelImg-master/labelImg_after_detection.py')

# for i in glob.glob(img_dir+'/*.jpg'):
#     print(i)
#     img = cv2.imread(i)
#     print(img.size)
#     img = cv2.resize(src=img, dsize=(416,416))
#     cv2.imwrite(i, img)
#


# img_dir = os.path.abspath('runs/detect/exp4/suggested_annot')
# predefined_class_file = os.path.abspath('runs/detect/exp4/suggested_annot/classes.txt')
# label_tool_path = os.path.abspath('utils/labelImg-master/labelImg_after_detection.py')
#
# # create txt in the same folder
#
# label_tool_path = '"' + label_tool_path + '"'
# img_dir = '"' + img_dir + '"'
# predefined_class_file = '"'+ predefined_class_file + '"'
# import subprocess
#
# cmd = 'python ' + label_tool_path + \
#       ' --image_dir ' + img_dir + \
#       ' --save_dir ' + img_dir + \
#       ' --predefined_classes_file '+ predefined_class_file
# s = subprocess.call(cmd, shell=True)
#
#


# source = r'C:\Users\Giulia Ciaramella\Desktop\v3d\cut-videos-ai\02_1_3noz_1internal.mp4'
# cap = cv2.VideoCapture(source)
# tot_frames = cv2.CAP_PROP_FRAME_COUNT
# fps = cap.get(cv2.CAP_PROP_FPS)
# cap.release()
#
# starting_point = '00:02'
# ending_point = '00:05'
#
# # transform in frames
# sp_m, sp_s = starting_point.split(':')
# st_sec = int(sp_m)*60 + int(sp_s)
# starting_frame = int(fps*st_sec)
#
# ep_m, ep_s = ending_point.split(':')
# et_sec = int(ep_m) * 60 + int(ep_s)
# ending_frame = int(fps * et_sec)
#
# print(starting_frame, ending_frame)