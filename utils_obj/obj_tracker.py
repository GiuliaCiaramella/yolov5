# method for tracking the object
import yaml
from utils_obj import obj_creation as Obj
import numpy as np
import json
import pandas as pd

def read_classes(yaml_file):
    with open(yaml_file, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    # save them in a dictionary
    inv_classes = dict(enumerate(conf['names'])) # 0: nozzle, 1:pipe etc
    classes = {v: k for k, v in inv_classes.items()} # nozzle: 0 ..
    return (classes, inv_classes)

class Tracker(object):
    def __init__(self,yaml_file):
        _ = read_classes(yaml_file)
        self.inv_classes = _[1]
        self.classes = list(_[0].keys())
        self.active_det = {k: [] for k in self.classes}
        self.deregistered_objs = {k: [] for k in self.classes} # objects that have been detected but are no more visible
        self.current_obj = Obj.Obj()
        self.distance_th = 0.065
        self.prev_dist = {k:[] for k in self.classes}
        self.total_objects = {k:0 for k in self.classes}

    def info(self, fps, save_dir):
        self.fps = fps
        self.save_dir = save_dir
        self.max_fr_deregister = int(fps)

    def update(self, nbox, bbox, cl, current_frame):
        # new detection comes

        self.current_frame = current_frame
        self.current_obj.nbox = nbox #  xywh normalized
        self.current_obj.bbox = bbox # xyxy not normalized
        # self.current_obj.historic_areas.append(2*(nbox[2]+nbox[3]))
        self.current_obj.cl = cl # 0,1,2 ..
        self.current_obj.label = self.inv_classes[cl]  # nozzle, pipe ..
        self.current_obj.centroid = (nbox[0], nbox[1]) #self.evaluate_centroid() # to evaluate centroids starting from nbox

        # process the input
        return self.process()

    def process(self):
        if not len(self.active_det[self.current_obj.label]):
            # new object
            self.update_field()
        else:
            lowest_distance = self.min_dist() # same object type at lowest dist, lower than Threshold; can also be None
            if lowest_distance:
                # update the object we already have in list of active detection
                self.current_obj.id = lowest_distance.id # just for printing
                self.update_old(lowest_distance)
                # self.prev_dist[self.current_obj.label].append(lowest_distance.dist)
            else:
                # other objects are not at min dist: new obj
                self.update_field()
        id = self.current_obj.id
        self.current_obj = Obj.Obj()  # clear object for next assignation
        self.deregister() # deregister all objects absent for more than a threshold
        return id

    def update_old(self, old):
        old.lframe = self.current_frame  # i am updating the object
        old.lvideo_sec = self.current_frame / self.fps
        old.tot_sec = old.lvideo_sec - old.ffvideo_sec
        old.centroid = self.current_obj.centroid
        #old.centroids.append(self.current_obj.centroid)
        old.bbox = self.current_obj.bbox
        old.nbox = self.current_obj.nbox
        old.historic_areas.append(old.nbox[2]*old.nbox[3])

    def update_field(self): # we enter here when we have a new object type or a new obj
        total = self.total_objects[self.current_obj.label]
        self.current_obj.id = self.current_obj.label + '_' + str(total+1)
        self.current_obj.fframe = self.current_frame
        self.current_obj.fvideo_sec = self.current_obj.fframe /  self.fps
        self.current_obj.lframe = self.current_frame
        self.current_obj.lvideo_sec = self.current_obj.lframe /  self.fps
        self.active_det[self.current_obj.label].append(self.current_obj)
        self.total_objects[self.current_obj.label] += 1 # update counting objects

    def min_dist(self):
        self.min_dist_obj = []
        for element in self.active_det[self.current_obj.label]:
           self.distance(element)
        # sort the objects with increasing distance
        if self.min_dist_obj:
            winning = sorted(self.min_dist_obj, key=lambda x: x.dist, reverse=False)[0]
            # I return the object with lowest distance
            return winning
        else:
            return None

    def distance(self, other):
        # method to evaluate the euclidean distance of objects
        current_c = self.current_obj.centroid
        other_c = other.centroid
        d = np.sqrt(np.sum((np.array(current_c) - np.array(other_c)) ** 2))
        if d < self.distance_th:
            other.dist = d
            self.min_dist_obj.append(other)
        else:
            pass

    def print_results(self):
        self.clean_short_periods()
        self.deregister(all=True) # first I deregister everything
        #print(self.deregistered_objs)
        result = self.convert_objects() # then, I convert each object in a dictionary to be json serializable

        # long and detailed summary
        with open(str(self.save_dir) + '/long.json', 'w') as j:
            json.dump(result, j, indent=4)
        j.close()

        # short summary of total objects
        with open(str(self.save_dir) + '/brief.json', 'w') as j:
            json.dump(self.total_objects, j, indent=4)
        j.close()

    def clean_short_periods(self):
        # remove objects that appeared for less than fps/2
        for k,v in list(self.deregistered_objs.items()):
            for o in v:
                if o.lframe -  o.fframe < self.fps:
                    self.deregistered_objs[k].remove(o)
                    self.total_objects[k] -= 1


    def convert_objects(self):
        result = {k:[] for k in list(self.deregistered_objs.keys())}
        for k, v in list(self.deregistered_objs.items()):
            for element in v:
                result[k].append(element.__dict__)
        return result



    def deregister(self, all=False):
        # when an object is not seen for more than self.max_fr_deregister = 10, it is deregistered
        # it is given to a new list that is not queried but needed only for final results
        if all: # all if we are at the end of the detection and I need to print all the results
            for k, l in list(self.active_det.items()):
                for element in l:
                    self.deregistered_objs[k].append(element)
        else:
            for k,l in list(self.active_det.items()):
                for element in l:
                    if element.lframe:
                        if element.lframe <= self.current_frame - self.max_fr_deregister:
                            self.deregistered_objs[element.label].append(element)
                            self.active_det[k].remove(element)




    def predict(self):
        # to predict the next position based on the previous position of centroids
        pass

    def dist_analysis(self): # for analysis
        for k in list(self.prev_dist.keys()):
            df = pd.DataFrame.from_dict(self.prev_dist[k], orient='columns')
            df.to_csv(str(self.save_dir)+'/'+k+'_dist.csv', index = False)