"""
1) vectorize dataset
2) apply k-means cluster to vectors
3) save centroids
4) new detection image:
    - vectorize
    - compare the distance of this vector with centroids
"""
import os
import cv2
import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt

img_path = r'C:\Users\Giulia Ciaramella\Desktop\v3d\new_data\new_aug_edges\img'
img_name = [f for f in os.listdir(img_path)]

c = 0
a = None
with Bar('vectorizing...', max=len(img_name)) as bar:
    for i in img_name:
        if c<10:
            source = cv2.imread(os.path.join(img_path, i))
            vec = source.reshape((-1, 1))
            # We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV.
            vectorized = np.float32(vec) # vector containing for each pixel 255 if white (edge) and 0 if black
             # concatenate over column bacause each row describe the same pixel
            if a is None:
                a = vectorized
            else:
                a = np.concatenate((a, vectorized), axis=1)
            c+=1
            bar.next()
        else:
            break

# a is an array of [W,H, images]
# now apply k means
# find centroid for each k
# save in json
print('here')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
attempts=10
comp,label,center=cv2.kmeans(a,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
print('here')

# output of kmeans:
# comp = compactness : It is the sum of squared distance from each point to their corresponding centers.
# labels : This is the label array (same as 'code' in previous article) where each element marked '0', '1'.....
# centers : This is array of centers of clusters

center = np.uint8(center)

# Now separate the data, Note the flatten()
A = a[label.flatten()==0]
B = a[label.flatten()==1]
# Plot the data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('White'),plt.ylabel('Black')
plt.show()