import yaml
import numpy as np
from scipy.spatial import distance
import pandas as pd
import warnings
import math

warnings.filterwarnings(action='ignore')
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
start_time = 5 * 60 + 10
end_time = 6 * 60 + 26
# input_video = r'F:\VivaDrive\v3d\BASF-F866-C200\F866-C200\123_0038.MOV'
# output_video = r'F:\VivaDrive\v3d\BASF-F866-C200\F866-C200\123_0038_cut.MOV'
# ffmpeg_extract_subclip(input_video, start_time, end_time, targetname=output_video)

