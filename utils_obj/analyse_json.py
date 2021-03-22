
# plot distribution of appearance duration in the video ---> for categories

# plot area of detected objects --> for categories --> for each object take the largest and the smallest
# --> in this way we can also analyze how small can be an object to be detected.
import json
import matplotlib.pyplot as plt

json_path = r'C:\Users\Giulia Ciaramella\PycharmProjects\yolov5\long.json'
with open (json_path, 'r') as j:
    data = json.load(j)
j.close()

#
areas = {c:[] for c in list(data.keys())}
apperance_duration = {c:[] for c in list(data.keys())}

for k,v in data.items():
    for o in v:
        #areas[k].append(max(o['historical_areas']))
        #apperance_duration[k].append(o['tot_sec'])
        a = o['nbox'][2]*o['nbox'][3]
        areas[k].append(a)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.hist(areas['int const'], bins = 10)
ax2.hist(areas['nozzle'], bins=10)
ax3.hist(areas['tube bundle'], bins=10)
ax4.hist(areas['pipe'], bins=10)

# plt.show()
fig.savefig('ciao.png')
# plt.show()


nbox = [0.4046875, 0.840625, 0.503125, 0.24375]
ar = nbox[2]*nbox[3]
print(ar)