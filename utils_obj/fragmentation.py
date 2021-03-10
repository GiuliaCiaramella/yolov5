import cv2
import time
import datetime
import os
from utils.general import increment_path
from pathlib import Path

def fragment_video(video_path, frame_rate=1):
        """
        :param video_path: where is the video to be fragmented
        :param frame_path: where you want your frames to be saved; default = video_path/fragmented/
        :param frame_rate: FPS, default = 3
        :param starting_point: second from where starting fragment; default = p
        :return:
        """
        video_name = os.path.basename(video_path)
        video_path = str(video_path)
        fp = os.path.join(video_path.replace(video_name, ''), 'frames_' + video_name.split('.')[-2])

        frame_path = increment_path(fp)
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)

        saved = 0
        passes = 0
        cap = cv2.VideoCapture(video_path)
        max_pass = 600
        # print(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # / cap.get(cv2.CAP_PROP_FPS)
        prev = 0
        i = -1
        while cap.isOpened():
            i+=1
            time_elapsed = time.time() - prev
            frame_exists, curr_frame = cap.read()
            if frame_exists:
                # cap.set(cv2.CAP_PROP_FPS, float(frame_rate))
                # cap.set(cv2.CAP_PROP_POS_MSEC, (i * 1000))  # added this line
                # if cap.get(cv2.CAP_PROP_POS_MSEC) > starting_point:
                if time_elapsed > 1. / frame_rate:
                    prev = time.time()
                    cv2.imwrite(os.path.join(frame_path, datetime.datetime.utcnow().strftime("%Y-%m-%d-%Hh-%Mm-%Ss-%fmics") + '_original.jpg'), curr_frame)
                    saved += 1
            else:
                passes += 1
            if passes > max_pass:
                break

        cap.release()
        cv2.destroyAllWindows()
        print('Done! %d images have been extracted. You can now procede with labeling :)' %saved)
        return(frame_path)


# fragment_video(r'C:\Users\Giulia Ciaramella\Desktop\v3d\cut-videos-ai\02_1_3noz_1internal.mp4')
