import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def crop_face_from_scene(image, bbox, scale):
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
    return region

def get_bbox(dat_path):
    with open(dat_path, 'r') as f:
        lines = f.readlines()
    x,y,w,h = [int(ele) for ele in lines[:4]]
    x=max(x,0)
    y=max(y,0)
    w=max(w,0)
    h=max(h,0)
    # print("{}, {}, {}, {}".format(x, y, w, h))
    return [int(x),int(y),int(w),int(h)]

def get_all_files(video_folder_path, suffix):
    all_file = []
    for f in os.listdir(video_folder_path):  #listdir返回文件中所有目录
        basename, ext = os.path.splitext(f)
        if (ext != suffix):
            continue
        f_name = os.path.join(video_folder_path, f)
        all_file.append(f_name)
    return all_file

def get_frameid(filename):
    elemlist = filename.split('_')
    if len(elemlist) >= 5:
        phone = elemlist[0]
        session = elemlist[1]
        people = elemlist[2]
        label = elemlist[3]
        frame = elemlist[4]

        frameid = "{}_{}_{}_{}_{}".format(phone, session, people, label, frame)
    else:
        frameid = filename
        print(filename)
    return frameid


def main(img_dir, dat_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    img_paths = get_all_files(img_dir, ".jpg")
    for img_path in img_paths:
        basename = os.path.basename(img_path)
        filename, extname = os.path.splitext(basename)
        frameid = get_frameid(filename)
        dat_path = os.path.join(dat_dir, frameid + ".dat")
        if not os.path.exists(dat_path):
            continue
        
        bbox = get_bbox(dat_path)

        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_scale = np.random.randint(12, 15)
        face_scale = face_scale/10.0

        crop = crop_face_from_scene(img, bbox, face_scale)

        dst_path = os.path.join(dst_dir, filename + ".jpg")
        cv2.imwrite(dst_path, crop)
        # print(dst_path)

def replayattack_main(img_dir, dat_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    img_paths = get_all_files(img_dir, ".jpg")
    for img_path in img_paths:
        basename = os.path.basename(img_path)
        filename, extname = os.path.splitext(basename)

        houzhui = filename.split('_')[-1]
        if houzhui == "depth":
            frameid = filename[:-6]
        else:
            frameid = filename
        
        print(frameid)
        dat_path = os.path.join(dat_dir, frameid + ".dat")
        if not os.path.exists(dat_path):
            continue
        
        bbox = get_bbox(dat_path)

        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_scale = np.random.randint(12, 15)
        face_scale = face_scale/10.0

        crop = crop_face_from_scene(img, bbox, face_scale)

        dst_path = os.path.join(dst_dir, filename + ".jpg")
        cv2.imwrite(dst_path, crop)
        # print(dst_path)

if __name__ == "__main__":
    # CASIA-FASD
    # TRAIN_FRAME_DIR = "/public/zzj/CASIA-FASD/train_release_frame"
    # TEST_FRAME_DIR = "/public/zzj/CASIA-FASD/test_release_frame"

    # TRAIN_DEPTH_DIR = "/public/zzj/CASIA-FASD/train_release_depth"
    # TEST_DEPTH_DIR = "/public/zzj/CASIA-FASD/test_release_depth"
    
    # TRAIN_DAT_DIR = "/public/zzj/CASIA-FASD/train_release_dat"
    # TEST_DAT_DIR = "/public/zzj/CASIA-FASD/test_release_dat"

    # TRAIN_FRAME_CROP_DIR = "/public/zzj/CASIA-FASD/train_release_frame_crop"
    # TEST_FRAME_CROP_DIR = "/public/zzj/CASIA-FASD/test_release_frame_crop"

    # TRAIN_DEPTH_CROP_DIR = "/public/zzj/CASIA-FASD/train_release_depth_crop"
    # TEST_DEPTH_CROP_DIR = "/public/zzj/CASIA-FASD/test_release_depth_crop"

    # Oulu-NPU
    # TRAIN_FRAME_DIR = "/public/zzj/Oulu-NPU/Train_frame"
    # DEV_FRAME_DIR = "/public/zzj/Oulu-NPU/Dev_frame"
    # TEST_FRAME_DIR = "/public/zzj/Oulu-NPU/Test_frame"

    # TRAIN_DEPTH_DIR = "/public/zzj/Oulu-NPU/Train_depth_real"
    # DEV_DEPTH_DIR = "/public/zzj/Oulu-NPU/Dev_depth_real"
    # TEST_DEPTH_DIR = "/public/zzj/Oulu-NPU/Test_depth_real"
    
    # TRAIN_DAT_DIR = "/public/zzj/Oulu-NPU/Train_dat"
    # DEV_DAT_DIR = "/public/zzj/Oulu-NPU/Dev_dat"
    # TEST_DAT_DIR = "/public/zzj/Oulu-NPU/Test_dat"

    # TRAIN_FRAME_CROP_DIR = "/public/zzj/Oulu-NPU/Train_frame_crop"
    # DEV_FRAME_CROP_DIR = "/public/zzj/Oulu-NPU/Dev_frame_crop"
    # TEST_FRAME_CROP_DIR = "/public/zzj/Oulu-NPU/Test_frame_crop"

    # TRAIN_DEPTH_CROP_DIR = "/public/zzj/Oulu-NPU/Train_depth_crop"
    # DEV_DEPTH_CROP_DIR = "/public/zzj/Oulu-NPU/Dev_depth_crop"
    # TEST_DEPTH_CROP_DIR = "/public/zzj/Oulu-NPU/Test_depth_crop"

    # Replay-Attack
    TRAIN_FRAME_DIR = "/public/zzj/Replay-Attack/train_frame"
    DEV_FRAME_DIR = "/public/zzj/Replay-Attack/devel_frame"
    TEST_FRAME_DIR = "/public/zzj/Replay-Attack/test_frame"

    TRAIN_DEPTH_DIR = "/public/zzj/Replay-Attack/train_depth"
    DEV_DEPTH_DIR = "/public/zzj/Replay-Attack/devel_depth"
    TEST_DEPTH_DIR = "/public/zzj/Replay-Attack/test_depth"
    
    TRAIN_DAT_DIR = "/public/zzj/Replay-Attack/train_dat"
    DEV_DAT_DIR = "/public/zzj/Replay-Attack/devel_dat"
    TEST_DAT_DIR = "/public/zzj/Replay-Attack/test_dat"

    TRAIN_FRAME_CROP_DIR = "/public/zzj/Replay-Attack/train_frame_crop"
    DEV_FRAME_CROP_DIR = "/public/zzj/Replay-Attack/devel_frame_crop"
    TEST_FRAME_CROP_DIR = "/public/zzj/Replay-Attack/test_frame_crop"

    TRAIN_DEPTH_CROP_DIR = "/public/zzj/Replay-Attack/train_depth_crop"
    DEV_DEPTH_CROP_DIR = "/public/zzj/Replay-Attack/devel_depth_crop"
    TEST_DEPTH_CROP_DIR = "/public/zzj/Replay-Attack/test_depth_crop"
    replayattack_main(TEST_DEPTH_DIR, TEST_DAT_DIR, TEST_DEPTH_CROP_DIR)