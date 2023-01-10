import os
import cv2
import threading

root_dir = "/public/zzj/Oulu-NPU"
proto_root_dir = os.path.join(root_dir, "Protocols")
protocols = ["Protocol_1", "Protocol_2", "Protocol_3", "Protocol_4"]
classes = ["Train", "Dev", "Test"]

PROTOCOL = 3

def parseOULUProtocol(str):
    label = str.split(',')[0]
    name = str.split(',')[1]
    return label, name

def getFrameInfo(clss, videoid, label):
    ret = []
    
    img_dir = os.path.join(root_dir, clss + "_frame")
    depth_dir = os.path.join(root_dir, clss + "_depth_real")
    dat_dir = os.path.join(root_dir, clss + "_dat")
    
    video_path = os.path.join(img_dir, videoid + ".avi")
    
    # 获取帧数
    frame_counter = 155
    # video_obj = cv2.VideoCapture(video_path)
    # frame_counter = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT))    # 总帧数
    # video_obj.release()
    
    # 获取每一帧info
    for i in range(frame_counter):
        frameid = videoid + "_frame%d" % i
        frame_path = os.path.join(img_dir, frameid + ".jpg")
        depth_path = os.path.join(depth_dir, frameid + "_depth.jpg")
        zero_path = os.path.join(depth_dir, "zero_depth.jpg")
        
        if label == 1:
            if os.path.exists(frame_path) and os.path.exists(depth_path):
                ret.append((frame_path, depth_path))
        else:
            if os.path.exists(frame_path):
                ret.append((frame_path, zero_path))
    
    return ret

def generate(protoc, clss):
    protocol_dir = os.path.join(proto_root_dir, protoc)
    for ii in range(1, 7):
        dst_dir = os.path.join(protocol_dir, "depth")
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        srcf_path = os.path.join(protocol_dir, clss + "_%d.txt" % ii)
        dstf_path = os.path.join(dst_dir, clss + "_%d.txt" % ii)
        # print(srcf_path)
        # print(dstf_path)

        srcf = open(srcf_path, 'r')
        dstf = open(dstf_path, 'w')

        line = srcf.readline()
        while line:
            label_str, videoid = parseOULUProtocol(line.strip('\n'))
            # print("{}, {}".format(label_str, videoid))

            if label_str == "+1":
                label = 1
            else:
                label = 0

            framesinfo = getFrameInfo(clss, videoid, label)
            for frameinfo in framesinfo:
                frame_path = frameinfo[0]
                depth_path = frameinfo[1]
                dstf.write(frame_path + " " + depth_path + " " + str(label) + "\n")

            line = srcf.readline()

        srcf.close()
        dstf.close()

if __name__ == "__main__":
    # for protocol 1 and 2
    if PROTOCOL == 0 or PROTOCOL == 1:
        protoc = protocols[PROTOCOL]
        protocol_dir = os.path.join(proto_root_dir, protoc)
        # print(protocol_dir)
        for clss in classes:
            srcf_path = os.path.join(protocol_dir, clss + ".txt")
            dst_dir = os.path.join(protocol_dir, "depth")
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            dstf_path = os.path.join(dst_dir, clss + ".txt")
            # print(srcf_path)
            # print(dstf_path)
            
            srcf = open(srcf_path, 'r')
            dstf = open(dstf_path, 'w')
            
            line = srcf.readline()
            while line:
                label_str, videoid = parseOULUProtocol(line.strip('\n'))
                # print("{}, {}".format(label_str, videoid))
                
                if label_str == "+1":
                    label = 1
                else:
                    label = 0
                
                framesinfo = getFrameInfo(clss, videoid, label)
                for frameinfo in framesinfo:
                    frame_path = frameinfo[0]
                    depth_path = frameinfo[1]
                    dstf.write(frame_path + " " + depth_path + " " + str(label) + "\n")
                
                line = srcf.readline()
                
            srcf.close()
            dstf.close()

    if PROTOCOL == 2 or PROTOCOL == 3:
        protoc = protocols[PROTOCOL]
        for clss in classes:
            generate(protoc, clss)