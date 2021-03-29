import matplotlib.pyplot as plt
import yaml
import os
import argparse
import time
from pathlib import Path
import cv2
import torch
import warnings
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from progress.bar import Bar

# from utils_obj.im_sim import Sim # qui
from utils_obj.obj_tracker import Tracker

warnings.filterwarnings(action='ignore')


def detect(save_img=False):
    source, start_frame, end_frame, weights, view_img, save_txt, imgsz, yaml_file = opt.source, \
                                                                                                    opt.start_frame, \
                                                                                                    opt.end_frame, \
                                                                                          opt.weights, opt.view_img, \
                                                                                          opt.save_txt, opt.img_size, \
                                                                                                           opt.yaml_file
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))



    # initialize Tracker and sim
    tracker = Tracker(yaml_file) # yaml file to read classes
    # sim = Sim(yaml_file=yaml_file) # qui

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # model_feat = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # initialize classifier for feature vector
    extract_features = False
    if extract_features:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()


    # Set Dataloader
    vid_path, vid_writer = None, None
    save_img = True
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    i = 0
    f = 0
    fvs = torch.Tensor([])
    with Bar('detection...', max=dataset.nframes) as bar:
        for path, img, im0s, vid_cap in dataset:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            duration = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) / vid_cap.get(cv2.CAP_PROP_FPS)
            # pass info to tracker
            if i == 0:
                tracker.info(fps = fps, save_dir = save_dir, video_duration = duration)
                # sim.info(fps = fps, save_dir = save_dir) # qui
                i=1

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # print(img.shape) # [1,3, W,H]
            # t1 = time_synchronized()

            if dataset.frame >= start_frame and dataset.frame<end_frame : # first frame is

                pred = model(img, augment=opt.augment)[0] # this is a tuple
                # pred = pred_total[0] # tensor [1,6552,9]
                # feat_tensor = pred_total[2] # tensor  [1,2808]
                # print(feat_tensor.shape)
                # if feat_tensor.shape[1]>1000:
                #     fvs = torch.cat((fvs,feat_tensor), 0 )


                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                # t2 = time_synchronized()

                # Apply second stage classifier Classifier
                # if classify:
                #     pred = apply_classifier(pred, modelc, img, im0s)

                # Apply classifier to retrieve feature vector


            else:
                f+=1
                pred = [torch.Tensor([])]

            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                clean_im = im0.copy()

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    l = []
                    lines = [] # to write results in txt if images are not similar
                    for *xyxy, conf, cls in reversed(det):
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        # take proprieties from the detection
                        nbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # xywh in normalized form
                        cl = int(cls.item())
                        bbox = torch.tensor(xyxy).view(1, 4)[0].tolist()
                        # pass proprieties to Tracker
                        id = tracker.update(nbox, bbox, cl, frame) # object put into the tracker

                        l = [int(cls.item())] + nbox
                        lines.append(l)

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, objectID = id,  label=label, color=colors[int(cls)], line_thickness=3) # label=label

                    # save detection in case the inspector wants to label the suggested images
                    # pass image to check similatiry
                    # can return 'sim' or 'not_sim'. If not_sim, we want to retrieve the detection too
                    # s_ = sim.new_im(clean_im, frame)  # qui
                    # if s_ == 'not_sim':
                    #     sim.save_detection(lines)

                    ## new way to extract features
                    if extract_features:
                        feat_tensor = apply_classifier(pred, modelc, img, im0s)
                        fvs = torch.cat((fvs, feat_tensor), 0)

                # save video
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

                vid_writer.write(im0)
                res = cv2.resize(im0, (416,416))
                cv2.imshow('frame', res)

                # cv2.imshow('frame', im0)
                cv2.waitKey(1)

            bar.next()


    # save fvs in a txt file
    import numpy as np
    try:
        vec_path = save_dir / './feat_vectors.txt'  # os.path.join(path,'feat_vectors.txt' )
        fvs_array = fvs.detach().cpu().numpy()
        mat = np.matrix(fvs_array)

        with open(vec_path, 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')
            f.close()

    except Exception as e:
        print(e)



    tracker.print_results()
    # sim.end() # qui

    if save_txt or save_img:
        print(f"Results saved to {save_dir}")


    # print('Mean time to assign id: ', np.mean(id_time))
    # print('With variance: ', np.var(id_time))

    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f)




if __name__ == '__main__':


    # from general conf, read which assets already exist and detection can be run
    general_conf = r'C:\Users\Giulia Ciaramella\PycharmProjects\E2E\general_conf.yaml'
    with open(general_conf) as file:
        d = yaml.full_load(file)
    file.close()

    assets = d['assets']
    i = False
    while not i:
        value = input("Please choose an asset. You can choose among: \n \033[1m%r\033[0m \n " % "   ".join(
            map(str, assets.keys())))
        if value not in list(assets.keys()):
            print('\033[91mError!\033[0m The asset you chose is not in the list.')
        else:
            i = True

    # read the path for the proper yaml file
    yaml_file = assets[value]
    with open(yaml_file) as file:
        current_yaml = yaml.full_load(file)
    file.close()

    # once read the proper yaml path, read weight file path
    wp = current_yaml['weight_file_path']
    weights = [os.path.join(wp, i) for i in os.listdir(wp) if i.endswith('pt')]
    weight = max(weights, key=os.path.getctime) # take the last weight

    # read conf lower limit adn img size of inference
    conf_th = current_yaml['detection_conf']
    size = current_yaml['detection_im_size']

    # How does the detector choose? goes on a video and press 'detect' or run detect and select the video?
    # source = r'F:\VivaDrive\v3d\fragmented_video_drone\pressure vessel\061_0038.mov'
    # source = r'C:\Users\Giulia Ciaramella\Desktop\v3d\cut-videos-ai\01_3internalc_360p.MOV'
    i = False
    while not i:
        source = input("Please select a video to process\n")
        if not os.path.exists(source):
            print('Sorry but the path does not exist.\n')
        else:
            i = True

    cap = cv2.VideoCapture(source)
    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # size = max(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    j = False
    while not j:
        cut = input(
            'Do you want to specify the starting and ending point of the video?\n[This can save time if the recording '
            'does not start in the interested environment]\n [y] or [n]?\n')
        if cut.lower() not in ['y', 'n']:
            print('Not valid input.')
        elif cut.lower()=='y':
            starting_point = input('Enter starting point as MM:SS (or "begin" to start from 0)\n')
            ending_point = input('Enter ending point as MM:SS (or "end" to process till the end)\n')

            if starting_point == 'begin':
                starting_frame = 1
            else:
                # transform in frames
                sp_m, sp_s = starting_point.split(':')
                st_sec = int(sp_m)*60 + int(sp_s)
                starting_frame = int(fps*st_sec)+1
            if ending_point == 'end':
                ending_frame = tot_frames
            else:
                ep_m, ep_s = ending_point.split(':')
                et_sec = int(ep_m) * 60 + int(ep_s)
                ending_frame = int(fps * et_sec)
            j = True
        elif cut.lower()== 'n':
            starting_frame = 1
            ending_frame = tot_frames
            j = True

    print(starting_frame, ending_frame)

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-file', nargs='+', type=str, default=yaml_file)
    parser.add_argument('--start_frame', nargs='+', type=str, default=starting_frame, help='first frame')
    parser.add_argument('--end_frame', nargs='+', type=str, default=ending_frame, help='first frame')

    parser.add_argument('--weights', nargs='+', type=str, default=weight, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=source, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=size, help='inference size (pixels)')

    parser.add_argument('--conf-thres', type=float, default=conf_th, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
            # import cProfile
            # import pstats
            # from pstats import SortKey
            # import io
            # import datetime

            # pr = cProfile.Profile()
            # pr.enable()
            # my_res = detect()
            # pr.disable()
            #
            # result = io.StringIO()
            # p = pstats.Stats(pr, stream=result).sort_stats(SortKey.CUMULATIVE)
            # # p = pstats.Stats(pr, stream=result).sort_stats(SortKey.TIME)
            #
            # p.print_stats()
            #
            # name = os.path.basename(Path(source)).split('.')[0]
            # with open(name+'_'+datetime.datetime.utcnow().strftime("%Y-%m-%d-%Hh-%Mm-%Ss")+'.txt', 'w+') as f:
            #     f.write(result.getvalue())
            # f.close()




# !python detect_original.py --weights weights/storage_tank_416img.pt --img 416 --conf 0.5 --source output-storage_tank_resized.mp4