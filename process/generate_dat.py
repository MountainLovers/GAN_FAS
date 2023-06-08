import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import location_data_pb2

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

def detect_face_mediapipe(img_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # For static images:
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img)

        # Draw face detections of each face.
        if not results.detections:
            return
        for detection in results.detections:
            location_data = detection.location_data
            if not location_data.HasField('relative_bounding_box'):
                return
            relative_bounding_box = location_data.relative_bounding_box
            x = relative_bounding_box.xmin
            w = relative_bounding_box.width
            
            y = relative_bounding_box.ymin
            h = relative_bounding_box.height
            
            x = int(x * width)
            y = int(y * height)
            w = int(w * width)
            h = int(h * height)
            
            # rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
            #       relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
            #       image_rows)
            # rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
            #       relative_bounding_box.xmin + relative_bounding_box.width,
            #       relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
            #       image_rows)
            # ret = [rect_start_point[0], rect_start_point[1], rect_end_point[0]-rect_start_point[0], rect_end_point[1]-rect_start_point[1]]
            ret = [x, y, w, h]
        return ret

def process_depth(src_dir, dst_dir):
    ferr = open(os.path.join(dst_dir, "err.txt"), 'w')
    cnt = 0
    frames = get_all_files(src_dir, ".jpg")
    for frame_path in frames:
        basename = os.path.basename(frame_path)
        path = os.path.dirname(frame_path)
        filename, extname = os.path.splitext(basename)
        
        dat_filename = filename + ".dat"
        dat_path = os.path.join(dst_dir, dat_filename)
        
        # 已经处理过
        if os.path.exists(dat_path):
            cnt += 1
            continue
        
        box_list = detect_face_mediapipe(frame_path)
        
        # 未检测到人脸
        if box_list is None:
            ferr.write(dat_path + "\n")
            cnt += 1
            continue
            
        with open(dat_path, 'w') as f:
            for ele in box_list:
                f.write(str(ele)+"\n")
        cnt += 1
        
        if (cnt % 100 == 0):
            print("{}/{} done.".format(cnt, len(frames)))
    ferr.close()


def casiafasd_main():
    TRAIN_FRAME_DIR = "/public/zzj/CASIA-FASD/train_release_frame"
    TEST_FRAME_DIR = "/public/zzj/CASIA-FASD/test_release_frame"

    TRAIN_DAT_DIR = "/public/zzj/CASIA-FASD/train_release_dat"
    TEST_DAT_DIR = "/public/zzj/CASIA-FASD/test_release_dat"

    src_dir = TEST_FRAME_DIR
    dst_dir = TEST_DAT_DIR

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    process_depth(src_dir, dst_dir)

def oulunpu_main():
    TRAIN_FRAME_DIR = "/public/zzj/CASIA-FASD/train_release_frame"
    DEV_FRAME_DIR = "/public/zzj/CASIA-FASD/test_release_frame"
    TEST_FRAME_DIR = "/public/zzj/CASIA-FASD/test_release_frame"

    TRAIN_DAT_DIR = "/public/zzj/CASIA-FASD/train_release_dat"
    TEST_DAT_DIR = "/public/zzj/CASIA-FASD/test_release_dat"

    src_dir = TEST_FRAME_DIR
    dst_dir = TEST_DAT_DIR

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    process_depth(src_dir, dst_dir)

def replayattack_main():
    TRAIN_FRAME_DIR = "/public/zzj/Replay-Attack/train_frame"
    DEV_FRAME_DIR = "/public/zzj/Replay-Attack/devel_frame"
    TEST_FRAME_DIR = "/public/zzj/Replay-Attack/test_frame"

    TRAIN_DAT_DIR = "/public/zzj/Replay-Attack/train_dat"
    DEV_DAT_DIR = "/public/zzj/Replay-Attack/devel_dat"
    TEST_DAT_DIR = "/public/zzj/Replay-Attack/test_dat"

    src_dir = TEST_FRAME_DIR
    dst_dir = TEST_DAT_DIR

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    process_depth(src_dir, dst_dir)

def masked_3dmad_main():
    FRAME_DIR = "/mnt/hdd.user/datasets/3DMAD/frames"

    DAT_DIR = "/mnt/hdd.user/datasets/3DMAD/dats"

    src_dir = FRAME_DIR
    dst_dir = DAT_DIR

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    process_depth(src_dir, dst_dir)

def masked_surf_main():
    FRAME_DIR = "/mnt/hdd.user/datasets/CASIA-SURF-3DMask/CASIA_SURF_3DMask/frames"

    DAT_DIR = "/mnt/hdd.user/datasets/CASIA-SURF-3DMask/CASIA_SURF_3DMask/dats"

    src_dir = FRAME_DIR
    dst_dir = DAT_DIR

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    process_depth(src_dir, dst_dir)

if __name__ ==  "__main__":
    masked_surf_main()