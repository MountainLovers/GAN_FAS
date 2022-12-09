import os
import cv2
import numpy as np
from util import util
import argparse
import threading

TRAIN_FILE_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Train_files"
TEST_FILE_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Test_files"
DEV_FILE_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Dev_files"

TRAIN_DAT_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Train_dat"
TEST_DAT_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Test_dat"
DEV_DAT_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Dev_dat"

TRAIN_NPY_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Train_npynew"
TEST_NPY_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Test_npynew"
DEV_NPY_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Dev_npynew"

TRAIN_RPPG_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Train_rppg"
TEST_RPPG_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Test_rppg"
DEV_RPPG_DIR = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Dev_rppg"

all_train_video = util.get_all_files(TRAIN_NPY_DIR, ".npy")
all_dev_video = util.get_all_files(TEST_NPY_DIR, ".npy")
all_test_video = util.get_all_files(TEST_NPY_DIR, ".npy")

def main():
    parser = argparse.ArgumentParser(description='Demo of argparse')

    parser.add_argument('--dir', type=int, default=0)
    parser.add_argument('--frame', type=int, default=64)

    args = parser.parse_args()
    print(args)

    itype = args.dir
    threhold = args.frame

    all_videos = [all_train_video, all_dev_video, all_test_video]
    dat_dirs = [TRAIN_DAT_DIR, DEV_DAT_DIR, TEST_DAT_DIR]
    npy_dirs = [TRAIN_NPY_DIR, DEV_NPY_DIR, TEST_NPY_DIR]
    typs = ["TRAIN", "DEV", "TEST"]

    videospath = all_videos[itype]
    dat_dir = dat_dirs[itype]
    npy_dir = npy_dirs[itype]
    
    errf = open(os.path.join(npy_dir, typs[itype] + "_" + str(threhold) + ".txt"), 'w')

    for videopath in videospath:
        video = np.load(videopath)   # [frame, width, height, channel]
        if video.shape[0] < threhold:
            errf.write(videopath + "\n")
        
    errf.close()

if __name__ == "__main__":
    main()