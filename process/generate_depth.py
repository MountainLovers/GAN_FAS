# In 3DDFA_V2

# before import, make sure FaceBoxes and Sim3DR are built successfully, e.g.,
# sh build.sh

import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.functions import draw_landmarks
from utils.render import render
from utils.depth import depth
import numpy as np

import matplotlib.pyplot as plt

def get_all_files(video_folder_path, suffix):
    all_file = []
    for f in os.listdir(video_folder_path):  #listdir返回文件中所有目录
        basename, ext = os.path.splitext(f)
        if (ext != suffix):
            continue
        f_name = os.path.join(video_folder_path, f)
        all_file.append(f_name)
    return all_file

# load config
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Init FaceBoxes and TDDFA, recommend using onnx flag
onnx_flag = True  # or True to use ONNX to speed up
if onnx_flag:
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    
    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX
    
    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
else:
    tddfa = TDDFA(gpu_mode=False, **cfg)
    face_boxes = FaceBoxes()

def get_depth(img, boxes, depth_path):
    # regress 3DMM params
        param_lst, roi_box_lst = tddfa(img, boxes)
        
        # reconstruct vertices and render depth
        dense_flag = True
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        depth(img, ver_lst, tddfa.tri, show_flag=False, wfp=depth_path, with_bg_flag=False)

def get_bbox(dat_path):
    with open(dat_path, 'r') as f:
        lines = f.readlines()
    x,y,w,h = [int(ele) for ele in lines[:4]]
    x=max(x,0)
    y=max(y,0)
    w=max(w,0)
    h=max(h,0)
    x1 = x
    x2 = x + w
    y1 = y
    y2 = y + h
    print("{}, {}, {}, {}".format(x1, y1, x2, y2))
    return [[int(x1),int(y1),int(x2),int(y2), 0.99]]

def casiafasd_main():
    TRAIN_FRAME_DIR = "/public/zzj/CASIA-FASD/train_release_frame"
    TEST_FRAME_DIR = "/public/zzj/CASIA-FASD/test_release_frame"

    TRAIN_DAT_DIR = "/public/zzj/CASIA-FASD/train_release_dat"
    TEST_DAT_DIR = "/public/zzj/CASIA-FASD/test_release_dat"

    TRAIN_DEPTH_DIR = "/public/zzj/CASIA-FASD/train_release_depth"
    TEST_DEPTH_DIR = "/public/zzj/CASIA-FASD/test_release_depth"
    

    src_dir = TRAIN_FRAME_DIR
    dat_dir = TRAIN_DAT_DIR
    dst_dir = TRAIN_DEPTH_DIR

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    framepaths = get_all_files(src_dir, ".jpg")
    ferr = open(os.path.join(dst_dir, "err_depth.txt"), 'w')
    cnt = 0
    for framepath in framepaths:
        basename = os.path.basename(framepath)
        path = os.path.dirname(framepath)
        filename, extname = os.path.splitext(basename)
        
        info = filename.split('_')
        
        # 生成dat路径
        dat_filename = filename + ".dat"
        dat_path = os.path.join(dat_dir, dat_filename)
        
        # 生成深度图路径
        depth_filename = filename + "_depth.jpg"
        depth_path = os.path.join(dst_dir, depth_filename)
        
        # 读图片
        img = cv2.imread(framepath)
        
        # 负样本
        if info[-2] != "1":
            cnt += 1
            if (cnt % 100 == 0):
                print("{}/{} done.".format(cnt, len(framepaths)))
            continue
            height=img.shape[0]
            width = img.shape[1]
            channel = img.shape[2]
            
            depth_img = np.zeros((height, width, channel))
            cv2.imwrite(depth_path, depth_img)
            
        
        # 正样本
        # 已经处理过
        # if os.path.exists(depth_path):
        #     cnt += 1
        #     if (cnt % 100 == 0):
        #         print("{}/{} done.".format(cnt, len(frames)))
        #     continue
        
        if not os.path.exists(dat_path):
            # 不存在dat数据
            ferr.write(dat_path + "\n")
            cnt += 1
            continue
        else:
            boxes = get_bbox(dat_path)
        
        get_depth(img, boxes, depth_path)
         
        cnt += 1
        
        if (cnt % 100 == 0):
            print("{}/{} done.".format(cnt, len(framepaths)))
    ferr.close()

def replayattack_main():
    TRAIN_FRAME_DIR = "/public/zzj/Replay-Attack/train_frame"
    DEV_FRAME_DIR = "/public/zzj/Replay-Attack/devel_frame"
    TEST_FRAME_DIR = "/public/zzj/Replay-Attack/test_frame"

    TRAIN_DAT_DIR = "/public/zzj/Replay-Attack/train_dat"
    DEV_DAT_DIR = "/public/zzj/Replay-Attack/devel_dat"
    TEST_DAT_DIR = "/public/zzj/Replay-Attack/test_dat"

    TRAIN_DEPTH_DIR = "/public/zzj/Replay-Attack/train_depth"
    DEV_DEPTH_DIR = "/public/zzj/Replay-Attack/devel_depth"
    TEST_DEPTH_DIR = "/public/zzj/Replay-Attack/test_depth"
    

    src_dir = TEST_FRAME_DIR
    dat_dir = TEST_DAT_DIR
    dst_dir = TEST_DEPTH_DIR

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    framepaths = get_all_files(src_dir, ".jpg")
    ferr = open(os.path.join(dst_dir, "err_depth.txt"), 'w')
    cnt = 0
    for framepath in framepaths:
        basename = os.path.basename(framepath)
        path = os.path.dirname(framepath)
        filename, extname = os.path.splitext(basename)
        
        info = filename.split('_')
        
        # 生成dat路径
        dat_filename = filename + ".dat"
        dat_path = os.path.join(dat_dir, dat_filename)
        
        # 生成深度图路径
        depth_filename = filename + "_depth.jpg"
        depth_path = os.path.join(dst_dir, depth_filename)
        
        # 读图片
        img = cv2.imread(framepath)
        
        # 负样本
        if info[-2] != "1":
            cnt += 1
            if (cnt % 100 == 0):
                print("{}/{} done.".format(cnt, len(framepaths)))
            continue
            height=img.shape[0]
            width = img.shape[1]
            channel = img.shape[2]
            
            depth_img = np.zeros((height, width, channel))
            cv2.imwrite(depth_path, depth_img)
            
        
        # 正样本
        # 已经处理过
        # if os.path.exists(depth_path):
        #     cnt += 1
        #     if (cnt % 100 == 0):
        #         print("{}/{} done.".format(cnt, len(frames)))
        #     continue
        
        if not os.path.exists(dat_path):
            # 不存在dat数据
            ferr.write(dat_path + "\n")
            cnt += 1
            continue
        else:
            boxes = get_bbox(dat_path)
        
        get_depth(img, boxes, depth_path)
         
        cnt += 1
        
        if (cnt % 100 == 0):
            print("{}/{} done.".format(cnt, len(framepaths)))
    ferr.close()

if __name__ == "__main__":
    replayattack_main()