import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a = [0,1,2,3,4,5,6,7,8,9]
plt.plot(np.arange(len(a)), a)
plt.show()


a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mat = np.matrix(a)
# with open('outfile.txt','wb') as f:
#     for line in mat:
#         np.savetxt(f, line, fmt='%.2f')

p = r'C:\Users\Giulia Ciaramella\Desktop\archive\flower_images\flower_images\centroid.txt'
with open(p,'r') as f:
    raw = []
    for line in f:
        r = line.split(' ')
        raw.append([float(i) for i in r])
    f.close()

b = np.asarray(raw)
print(b)