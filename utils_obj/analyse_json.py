
# plot distribution of appearance duration in the video ---> for categories

# plot area of detected objects --> for categories --> for each object take the largest and the smallest
# --> in this way we can also analyze how small can be an object to be detected.
import json
import os
import numpy as np
import matplotlib.pyplot as plt

p_general = r'C:\Users\Giulia Ciaramella\PycharmProjects\yolov5\runs\detect'
exps = [os.path.join(p_general, i) for i in os.listdir(p_general)]
p = max(exps, key=os.path.getctime)
print('Analysing directory ', p)


with open (os.path.join(p, 'long.json'), 'r') as j:
    data = json.load(j)
j.close()

#
areas = {c:[] for c in list(data.keys())}
apperance_duration = {c:[] for c in list(data.keys())}
max_dimension = {c:[] for c in list(data.keys())}

# extract info on max dimension in a object, total duration in the video of an object
for k,v in data.items():
    for o in v:
        if o['tot_sec'] != None:
            apperance_duration[k].append(o['tot_sec'])
        a = max(o['past_appearance'], key = lambda x: x['width']*x['height'])
        if a is not None:
            areas[k].append(a['width']*a['height'])
        d = max( o['past_appearance'], key = lambda x: x['max_dim'])
        if d is not None:
            max_dimension[k].append(d['max_dim'])
            o['max_app_dim'] = d


# to plot
figures = ['areas.png', 'appearance_duration.png','dimensions.png']
datas = [areas, apperance_duration, max_dimension]
color = ['orange', 'blue', 'green']
t = 0
for d in datas:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax = [ax1, ax2, ax3, ax4]
    i = 0
    for k in list(d.keys()):
        try:
            ax[i].hist(d[k], bins = 20, color=color[t], alpha=0.5)
            ax[i].title.set_text(k)
        except Exception:
            pass
        i+=1
    title = figures[t].replace('.png', '')
    fig.suptitle(title.replace('_', ' '))
    fig.tight_layout()
    fig.savefig(os.path.join(p,figures[t]))
    t+=1

def skipped_frames():
    # skipped frames can be a sign of 'missed detections' or 'false positives'
    for cl, cl_items in data.items():
        for item in cl_items:
            frames = (item['lframe']-item['fframe'] )
            registered_frames = (len(item['past_appearance'])-1)
            # print('item: ', item['id'], ' from frame: ', item['fframe'], ' to frame ', item['lframe'],
            #       'skipped frames: ', frames - registered_frames )
            sk = frames - registered_frames
            if sk > 0:
                all_frames = np.arange(item['fframe'], item['lframe']+1, dtype='float64')
                present = [i['frame'] for i in item['past_appearance'] if i['frame'] in all_frames]
                # missing = [i for i in all_frames if i not in present]
                missing = list(set(all_frames).difference(present))
                item['skipped_frame'] = missing
                # print('\t ---> ', missing)
            else:
                item['skipped_frame'] = 0

            # print(sk if sk>0 else None)

def notable_objects(min_dim=0.3, min_duration=0, sk_th = 10):
    # notable objects are thos that respect the filters (parameters of the function)
    # saves new json if notable objects are found
    results = []
    for cl, cl_items in data.items():
        for item in cl_items:
            d = max(item['past_appearance'], key=lambda x: x['max_dim'])
            duration = item['tot_sec']
            if type(item['skipped_frame']) == list:
                sk = len(item['skipped_frame'])
            else:
                sk = item['skipped_frame']
            if d['max_dim'] >= min_dim and duration >= min_duration and sk<= sk_th:
                results.append(item)
    if results:
        with open(os.path.join(p, 'filtered_results.json'), 'w') as j:
            json.dump(results, j, indent=4)
        j.close()
        print('%d items satisfy the filtering conditions.' %len(results))
    else:
        print('No item satisfied the filtering conditions.')

skipped_frames()
notable_objects()