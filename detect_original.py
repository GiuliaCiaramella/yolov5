import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import yaml
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = False #True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        print(pred)
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
         #   print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
            #res = cv2.resize(im0, (416, 416))
            # cv2.imshow('frame', res)
            cv2.imshow('frame', im0)
            cv2.waitKey(1)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    #print(f'Done. ({time.time() - t0:.3f}s)')


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
    weights = [os.path.join(wp, i) for i in os.listdir(wp) if i.endswith('pt')]  # * means all if need specific format then *.csv
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
    cap.release()

    j = False
    while not j:
        cut = input(
            'Do you want to specify the staring and ending point of the video?\n[This can save time if the recording'
            'does not start in the interested environment]\n [y] or [n]?\n')
        if cut.lower() not in ['y', 'n']:
            print('Not valid input.')
        elif cut.lower()=='y':
            starting_point = input('Enter starting point as MM:SS\n')
            ending_point = input('Enter ending point as MM:SS\n')

            # transform in frames
            sp_m, sp_s = starting_point.split(':')
            st_sec = int(sp_m)*60 + int(sp_s)
            starting_frame = int(fps*st_sec)+1

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

    times = []
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            # detect()
            import cProfile
            # cProfile.run('detect()', 'restats')
            import pstats
            from pstats import SortKey
            import io

            pr = cProfile.Profile()
            pr.enable()
            my_res = detect()
            pr.disable()

            result = io.StringIO()
            p = pstats.Stats(pr, stream=result).sort_stats(SortKey.CUMULATIVE)
            p.print_stats()

            name = os.path.basename(Path(source)).split('.')[0]
            with open(name + '_orig.txt', 'w+') as f:
                f.write(result.getvalue())
            f.close()

