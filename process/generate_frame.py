import os
import cv2
import math
import numpy as np

def get_all_files(video_folder_path, suffix):
    all_file = []
    for f in os.listdir(video_folder_path):  #listdir返回文件中所有目录
        basename, ext = os.path.splitext(f)
        if (ext != suffix):
            continue
        f_name = os.path.join(video_folder_path, f)
        all_file.append(f_name)
    return all_file

def extract_images_from_video(video_path, dst_dir, frame_nums, videoid=None):
    video_obj = cv2.VideoCapture(video_path)
    print(video_obj.isOpened())
    basename = os.path.basename(video_path)
    path = os.path.dirname(video_path)
    filename, extname = os.path.splitext(basename)
    
    width = int(video_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))     
    fps = int(round(video_obj.get(cv2.CAP_PROP_FPS)))    # 帧率
    frame_counter = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT))    # 总帧数

    print("帧率: %d, 总帧数: %d, 分辨率: width * height %d * %d" % (fps, frame_counter, width, height))
    
    frame_nums = min(frame_nums, frame_counter)
    count = 0
    success = True
    
    # frames = np.zeros((frame_nums, height, width, 3), dtype='uint8')
#     print(frames.shape)
    while (video_obj.isOpened and success != False and count < frame_nums):
        success, image = video_obj.read()
        if (success != False):
            if (type(image) != None):
                if videoid == None:
                    videoid = filename
                cv2.imwrite(os.path.join(dst_dir, videoid + "_frame%d.jpg" % count), image)
                # frames[count] = image
                count += 1
                
#     print(count)
    video_obj.release()
    cv2.destroyAllWindows()
    # return frames

def casiafasd_main():
    TRAIN_FILE_DIR = "/public/zzj/CASIA-FASD/train_release"
    TEST_FILE_DIR = "/public/zzj/CASIA-FASD/test_release"

    TRAIN_FRAME_DIR = "/public/zzj/CASIA-FASD/train_release_frame"
    TEST_FRAME_DIR = "/public/zzj/CASIA-FASD/test_release_frame"

    def casiafasd_name(peopleid, videoid):
        # 根据casiafasd规则获取videoid, format: phone_session_people_label
        phone = -1
        label = -1
        if len(videoid) == 1:
            if int(videoid) % 2 == 1:
                phone = 1
                t = int(videoid) // 2
                label = t + 1
            else:
                phone = 0
                t = int(videoid) // 2
                label = t
        else:
            phone = 2
            t = int(videoid[-1])
            label = t
        
        name = "%s_%s_%s_%s" % (phone, "0", peopleid, label)
        return name

    src_dir = TRAIN_FILE_DIR
    dst_dir = TRAIN_FRAME_DIR

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for peopleid in os.listdir(src_dir):
        people_dir = os.path.join(src_dir, peopleid)
        videos = get_all_files(people_dir, ".avi")
        for video_path in videos:
            basename = os.path.basename(video_path)
            filename, extname = os.path.splitext(basename)
            # print(filename)
            videoid = casiafasd_name(peopleid, filename)
            print(video_path + ": " + videoid)
            extract_images_from_video(video_path, dst_dir, 1000, videoid)

def oulunpu_main():
    TRAIN_FILE_DIR = "/public/zzj/Oulu-NPU/Train_files"
    DEV_FILE_DIR = "/public/zzj/Oulu-NPU/Dev_files"
    TEST_FILE_DIR = "/public/zzj/Oulu-NPU/Test_files"

    TRAIN_FRAME_DIR = "/public/zzj/Oulu-NPU/Train_frame"
    DEV_FRAME_DIR = "/public/zzj/Oulu-NPU/Dev_frame"
    TEST_FRAME_DIR = "/public/zzj/Oulu-NPU/Test_frame"

    src_dir = TEST_FILE_DIR
    dst_dir = TEST_FRAME_DIR

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    videos = get_all_files(src_dir, ".avi")
    for video_path in videos:
        basename = os.path.basename(video_path)
        filename, extname = os.path.splitext(basename)
        print(filename)
        # print(video_path + ": " + filename)
        extract_images_from_video(video_path, dst_dir, 1000)

def replayattack_main(clssidx=0):
    ROOT_DIR = "/public/zzj/Replay-Attack"
    clss = ["train", "devel", "test"]
    labeldir = ["attack", "real"]
    cameradir = ["fixed", "hand"]

    video_paths = []
    for ld in labeldir:
        if ld == "attack":
            for cd in cameradir:
                finaldir = os.path.join(ROOT_DIR, clss[clssidx], ld, cd)
                tmp_paths = get_all_files(finaldir, ".mov")
                for i, item in enumerate(tmp_paths):
                    # 前面加fixed/hand，后面加label，0代表fake
                    tmp_paths[i] = [item, cd, 0]
                video_paths += tmp_paths
        else:
            finaldir = os.path.join(ROOT_DIR, clss[clssidx], ld)
            tmp_paths = get_all_files(finaldir, ".mov")
            for i, item in enumerate(tmp_paths):
                tmp_paths[i] = [item, "real", 1]
            video_paths += tmp_paths
    
    print(len(video_paths))

    dst_dir = os.path.join(ROOT_DIR, clss[clssidx] + "_frame")
    # print(dst_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for video_item in video_paths:
        video_path = video_item[0]
        info = video_item[1]
        label = str(video_item[2])

        print(video_path)
        basename = os.path.basename(video_path)
        filename, extname = os.path.splitext(basename)
        # print(filename)
        extract_images_from_video(video_path, dst_dir, 1000, filename + "_" + info + "_" + label)

if __name__ == "__main__":
    replayattack_main(2)

    