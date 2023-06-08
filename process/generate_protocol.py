# For rPPG text LOOCV method

import os
import cv2
import numpy as np
import threading
import random

root_dir = "/mnt/hdd.user/datasets/CASIA-SURF-3DMask/CASIA_SURF_3DMask"
proto_dir = os.path.join(root_dir, "Protocols")
npy_dir = os.path.join(root_dir, "npy")
rppg_dir = os.path.join(root_dir, "rppg")
classes = ["Train", "Dev", "Test"]

def int2str(x):
    if (x < 10):
        return "0" + str(x)
    return str(x)

def get_frames(npypath):
    x = np.load(npypath)
    return x.shape[0]

def get_all_files(video_folder_path, suffix):
    all_file = []
    for f in os.listdir(video_folder_path):  #listdir返回文件中所有目录
        basename, ext = os.path.splitext(f)
        if (ext != suffix):
            continue
        f_name = os.path.join(video_folder_path, f)
        all_file.append(f_name)
    return all_file

def get_all_subjects():
    def get_subject_id(filepath):
        filename = filepath.split('/')[-1]
        return filename.split('_')[0]

    all_files = get_all_files(npy_dir, ".npy")
    all_subjects = []

    for filepath in all_files:
        sid = get_subject_id(filepath)
        if sid not in all_subjects:
            all_subjects.append(sid)
    
    return all_subjects
    

windows = 64
stride = 15

subjects_num = 48
condition_num = 6

if __name__ == "__main__":
    all_subjects = get_all_subjects()

    for test_people in range(1, subjects_num + 1):
        train_f = open(os.path.join(proto_dir, "Train_"+str(test_people)+".txt"), "w")
        dev_f = open(os.path.join(proto_dir, "Dev_"+str(test_people)+".txt"), "w")
        test_f = open(os.path.join(proto_dir, "Test_"+str(test_people)+".txt"), "w")
        err_f = open(os.path.join(proto_dir, "Err_"+str(test_people)+".txt"), "w")

        allpeople = []
        for i in range(1, subjects_num + 1):
            if (i == test_people):
                continue
            allpeople.append(i)
        
        trainpeople = random.sample(allpeople, 8)

        for people in range(1, subjects_num + 1):
            for condition in range(1, condition_num + 1):
                for label in range(0, 4):
                    npyname = "{}_{}_{}_yuv32.npy".format(all_subjects[people - 1], str(condition), str(label))
                    npypath = os.path.join(npy_dir, npyname)
                    rppgname = "{}_{}_{}_yuv8.npy".format(all_subjects[people - 1], str(condition), str(label))
                    rppgpath = os.path.join(rppg_dir, rppgname)
                    
                    flag = True
                    if not os.path.exists(npypath):
                        err_f.write(npypath + "\n")
                        flag = False
                        continue
                    
                    if label == 0 and not os.path.exists(rppgpath):
                        err_f.write(rppgpath + "\n")
                        flag = False
                        continue

                    framesnum = get_frames(npypath)
                    if framesnum < windows:
                        err_f.write(npypath + " frames: " + str(framesnum) + "\n")
                        flag = False
                        continue
                    
                    if not flag:
                        continue
                    
                    if test_people == people:
                        write_f = test_f
                    else:
                        if people in trainpeople:
                            write_f = train_f
                        else:
                            write_f = dev_f   

                    st = 0
                    ed = windows

                    while (ed < framesnum):
                        if flag:
                            if label == 0:
                                wlabel = "1"
                            else:
                                wlabel = "0"
                            write_f.write(npypath + " " + rppgpath + " " + str(st) + " " + str(ed) + " " + wlabel + "\n")
                            st += stride
                            ed += stride
                        if stride == 0:
                            break
                    
        train_f.close()
        dev_f.close()
        test_f.close()