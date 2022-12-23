import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import threading
import csv
import argparse

def get_all_files(video_folder_path, suffix):
    all_file = []
    for f in os.listdir(video_folder_path):  #listdir返回文件中所有目录
        basename, ext = os.path.splitext(f)
        if (ext != suffix):
            continue
        f_name = os.path.join(video_folder_path, f)
        all_file.append(f_name)
    return all_file

def get_origin_frames(video_path):
    try:
        video_obj = cv2.VideoCapture(video_path)
    except:
        print("{} cv2 read error".format(video_path))
    width = int(video_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(video_obj.get(cv2.CAP_PROP_FPS))) # 帧率
    frame_counter = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT)) # 总帧数
#     print("帧率: %d, 总帧数: %d, 分辨率: width * height %d * %d" % (fps, frame_counter, width, height))
    frame_nums = frame_counter
    count = 0
    success = True
    
    frames = np.zeros((frame_nums, height, width, 3), dtype='uint8')
    framesyuv = np.zeros((frame_nums, height, width, 3), dtype='uint8')
#     print(frames.shape)
    while (video_obj.isOpened and success != False and count < frame_nums):
        success, image = video_obj.read()
        if (success != False):
            if (type(image) != None):
                frames[count] = image
                
                imageyuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                framesyuv[count] = imageyuv
                
                count += 1
#     print(count)
    video_obj.release()
    cv2.destroyAllWindows()
    return frames, framesyuv


def crop_face(image, bbox, scale):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    w_img, h_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale = scale * w
    h_scale = scale * h
    y1 = y_mid - w_scale / 2.0
    x1 = x_mid - h_scale / 2.0
    y2 = y_mid + w_scale / 2.0
    x2 = x_mid + h_scale / 2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)

    region=image[y1:y2,x1:x2]
    # region=image[x1:x2,y1:y2]
    return region

# 读取.face文件，获得bbox
def get_bbox(dat_dir, videoid):
    dat_path = os.path.join(dat_dir, videoid+".face")
    bbox = []
    if not os.path.exists(dat_path):
        print("Error! {} dat not exist! {}".format(videoid, dat_path))
        return bbox
        
    with open(dat_path, 'r') as f:
        lines = f.readlines()

    ii = 0
    for line in lines:
        line = line.split(",")
        i, left, top, right, bottem = [float(ele) for ele in line[:5]]
        while ii < i:
            bbox.append([])
            ii += 1
        x = left
        y = top
        w = right - left
        h = bottem - top
        bbox.append([x,y,w,h])
        ii += 1
    return bbox

def get_cropped_frames(origin_frames, bboxes, videoid, height, width):

    frame_nums = origin_frames.shape[0]
    cropped_frames = np.zeros((frame_nums, height, width, 3), dtype='uint8')

    for idx, image in enumerate(origin_frames):
        # face不存在
        if (idx >= len(bboxes) or len(bboxes[idx]) == 0):
            print(len(bboxes[idx]))
            print("Error! {} {} frame dat not exist!".format(videoid, idx))
            continue
        
        bbox = bboxes[idx]

        cropped_image = crop_face(image, bbox, 1.4)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2YUV)
        resized_image = cv2.resize(cropped_image, (height, width), interpolation=cv2.INTER_AREA)
        cropped_frames[idx] = resized_image

    return cropped_frames


def video2data(videopath, dat_dir, height, width):
    basename = os.path.basename(videopath)
    basedir = os.path.dirname(videopath)

    videoid, extname = os.path.splitext(basename)

    origin_frames, origin_frames_yuv = get_origin_frames(videopath)
    # print("{} | origin_frames.shape: {}".format(videoid, origin_frames.shape))
#     show_img(origin_frames, 30)

    bboxes = get_bbox(dat_dir, videoid)
    # print("{} | len bboxes: {}".format(videoid, len(bboxes)))


    cropped_frames = get_cropped_frames(origin_frames, bboxes, videoid, height, width)
    cropped_frames_yuv = get_cropped_frames(origin_frames_yuv, bboxes, videoid, height, width)
#     show_img(cropped_frames, 30)
    return cropped_frames, cropped_frames_yuv


# test
def show_img(image, idx=-1):
    plt.figure(1)
    if idx == -1:
        for img in image:
            plt.imshow(img)
            plt.show()
    else:
        plt.imshow(image[idx])
        plt.show()

def process(videospath, dat_dir, npy_dir, w, h):
    totalcnt = 0
    realcnt = 0
    shouldcnt = len(videospath)
    for videopath in videospath:
        basename = os.path.basename(videopath)
        basedir = os.path.dirname(videopath)
        videoid, extname = os.path.splitext(basename)
        
        print("{} | now total: {}, now real: {}, should total: {}".format(videoid, totalcnt, realcnt, shouldcnt))
        
        rgbname = videoid+"_rgb"+str(w)
        yuvname = videoid+"_yuv"+str(w)
        if os.path.exists(os.path.join(npy_dir, rgbname+".npy")) and os.path.exists(os.path.join(npy_dir, yuvname+".npy")):
            realcnt += 1
            totalcnt += 1
            continue

        if True:
            outputrgb, outputyuv = video2data(videopath, dat_dir, w, h)
            np.save(os.path.join(npy_dir, rgbname), outputrgb)
            np.save(os.path.join(npy_dir, yuvname), outputyuv)
            realcnt += 1
        totalcnt += 1


BASE_DIR = "/mnt/hdd.user/datasets/FAS/MSU-MFSD/scene01"

ATTACK_FILE_DIR = os.path.join(BASE_DIR, "attack")
REAL_FILE_DIR = os.path.join(BASE_DIR, "real")

ATTACK_NPY_DIR = os.path.join(BASE_DIR, "attack_npy")
REAL_NPY_DIR = os.path.join(BASE_DIR, "real_npy")

all_attack_video = get_all_files(ATTACK_FILE_DIR, ".mov") + get_all_files(ATTACK_FILE_DIR, ".mp4")
all_real_video = get_all_files(REAL_FILE_DIR, ".mov") + get_all_files(REAL_FILE_DIR, ".mp4")

def main():
    parser = argparse.ArgumentParser(description='Demo of argparse')

    parser.add_argument('--dir', type=int, default=0)
    parser.add_argument('--threads', type=int, default=5)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--height', type=int, default=32)


    args = parser.parse_args()
    print(args)

    itype = args.dir
    threads_num = args.threads
    w = args.width
    h = args.height

    threads_list = []

    all_videos = [all_attack_video, all_real_video]
    dat_dirs = [ATTACK_FILE_DIR, REAL_FILE_DIR]
    npy_dirs = [ATTACK_NPY_DIR, REAL_NPY_DIR]

    videospath = all_videos[itype]
    dat_dir = dat_dirs[itype]
    npy_dir = npy_dirs[itype]

    avgcnt = int(len(videospath) / threads_num)
    print(len(videospath))
    print(avgcnt)
    videoipath = [videospath[i:i+avgcnt] for i in range(0,len(videospath),avgcnt)]
    print(videoipath)
    
    for ii in range(threads_num):
        t = threading.Thread(target=process, args=(videoipath[ii], dat_dir, npy_dir, w, h,))
        threads_list.append(t)
        t.start()

    for t in threads_list:
        t.join()

if __name__ == "__main__":
    main()
