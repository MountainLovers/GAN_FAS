
import os
import cv2
import threading

root_dir = "/public/zzj/CASIA-FASD"
proto_root_dir = os.path.join(root_dir, "Protocols")
TRAIN_FRAME_DIR = os.path.join(root_dir, "train_release_frame")
TEST_FRAME_DIR = os.path.join(root_dir, "test_release_frame")
TRAIN_DEPTH_DIR = os.path.join(root_dir, "train_release_depth")
TEST_DEPTH_DIR = os.path.join(root_dir, "test_release_depth")
TRAIN_DAT_DIR = os.path.join(root_dir, "train_release_dat")
TEST_DAT_DIR = os.path.join(root_dir, "test_release_dat")

def parseCASIAName(s):
    basename = os.path.basename(s)
    filename, ext = basename.split('.')
    label = filename.split('_')[-2]
    phone = filename.split('_')[0]
    return label, phone, filename

def getFrameInfo(clss, videoid, label, phone):
    ret = []

    if clss == "Train":
        img_dir = TRAIN_FRAME_DIR
        depth_dir = TRAIN_DEPTH_DIR
        dat_dir = TRAIN_DAT_DIR
    
    if clss == "Test":
        img_dir = TEST_FRAME_DIR
        depth_dir = TEST_DEPTH_DIR
        dat_dir = TEST_DAT_DIR
        
    
    frameid = videoid
    frame_path = os.path.join(img_dir, frameid + ".jpg")
    depth_path = os.path.join(depth_dir, frameid + "_depth.jpg")
    dat_path = os.path.join(dat_dir, frameid + ".txt")
    zero_path = os.path.join(depth_dir, "zero_depth_"+str(phone)+".jpg")
    
    if label == 1:
        if os.path.exists(frame_path) and os.path.exists(depth_path):
            ret.append((frame_path, depth_path, dat_path))
    else:
        if os.path.exists(frame_path):
            ret.append((frame_path, zero_path, dat_path))
    
    return ret

def get_all_files(video_folder_path, suffix):
    all_file = []
    for f in os.listdir(video_folder_path):  #listdir返回文件中所有目录
        basename, ext = os.path.splitext(f)
        if (ext != suffix):
            continue
        f_name = os.path.join(video_folder_path, f)
        all_file.append(f_name)
    return all_file

def generate(clss, frame_dir, depth_dir, dst_dir):
    frame_paths = get_all_files(frame_dir, ".jpg")
    depth_paths = get_all_files(depth_dir, ".jpg")

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
        
    dstf_path = os.path.join(dst_dir, clss + ".txt")

    dstf = open(dstf_path, 'w')

    for frame_path in frame_paths:
        label_str, phone, videoid = parseCASIAName(frame_path)

        if label_str == "1":
            label = 1
        else:
            label = 0

        framesinfo = getFrameInfo(clss, videoid, label, phone)
        for frameinfo in framesinfo:
            frame_path = frameinfo[0]
            depth_path = frameinfo[1]
            dat_path = frameinfo[2]
            dstf.write(frame_path + " " + depth_path + " " + str(label) + "\n")

    dstf.close()

if __name__ == "__main__":
    generate("Test", TEST_FRAME_DIR, TEST_DEPTH_DIR, proto_root_dir)