# specification for an object
class Obj(object):
    def __init__(self):
        # never change:
        self.cl = None
        self.label = ''
        self.id = ''
        self.number = 0
        self.fframe = None # similar to self.fvideo_sec but for frames
        self.fvideo_sec = None # first second where it appears in the video
        # change as the object moves:
        self.lframe = None
        self.lvideo_sec = None # last second of video where it appears
        self.nbox = None  # xywh in normalized form
        self.bbox = None
        self.centroid = None
        self.dist = 1000
       # self.centroids = []
