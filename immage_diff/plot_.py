import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
"""
in blue, the points that belongs to the dataset or those that are seen as new point that should  be added to the dataset
in grean the current max vector
in red x, the element that are rejected (not put in the data)
"""


rs = np.random.RandomState(42)
X = rs.rand(50,2)
rejected = np.array([])

plt.axis([min(X[:,0])-5, max(X[:,0])+5, min(X[:,1])-5, max(X[:,1])+5])
plt.xlabel('x')
plt.ylabel('y')
filenames = []
frames = []
for i in range(50):
    if rejected.size:
        plt.scatter(rejected[:,0], rejected[:,1], marker='x', s=18, color='red')

    # plot data point as blue
    plt.scatter(X[:, 0], X[:, 1], s=16, color='blue')

    # get centroid
    centroid = np.mean(X, axis = 0)

    # get current point at max distance
    norms = {np.linalg.norm(centroid - k): k for k in X}
    max_dist = max(norms.keys())
    current_max = norms[max_dist]
    plt.scatter(current_max[0], current_max[1], s=64, color='green')

    # plot connection lines
    for x in X:
        plt.plot([x[0], centroid[0]], [x[1], centroid[1]], color='grey', linewidth=0.2)
    plt.scatter(centroid[0], centroid[1], color='orange', marker='s', s=20)


    # new point:
    new_point = np.random.rand(1, 2) * np.random.randint(low=0, high=10) / 3
    plt.scatter(new_point[:, 0], new_point[:, 1], color='black', s=16)

    # here the metric
    new_dist = np.linalg.norm(new_point - centroid)
    if new_dist > max_dist:
        X = np.concatenate((X,new_point))
    else:
        # rejected = rejected.concatenate((rejected,new_point), axis=0) if rejected.size else new_point
        rejected = np.vstack([rejected, new_point]) if rejected.size else new_point




    plt.pause(0.1)
    fn = str(i)+'.png'
    plt.savefig(fn)
    filenames.append(fn)
    frames.append(imageio.imread(fn))
    plt.clf()


plt.close()
imageio.mimsave('exportname.gif', frames, duration=0.8)

for filename in set(filenames):
    os.remove(filename)