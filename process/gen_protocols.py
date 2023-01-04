import os
import cv2
import threading
import argparse
import numpy as np

root_dir = "/public/zzj/Oulu-NPU"
proto_root_dir = "/public/zzj/Oulu-NPU/Protocols"
protocols = ["Protocol_1", "Protocol_2", "Protocol_3", "Protocol_4"]
classes = ["Train", "Dev", "Test"]
npy_dirs = ["Train_npynew", "Dev_npynew", "Test_npynew"]
rppg_dirs = ["Train_rppg", "Dev_rppg", "Test_rppg"]

windows = 64
stride = 15

def parseOULUProtocol(str):
    label = str.split(',')[0]
    name = str.split(',')[1]
    return label, name

def get_frames(npypath):
    x = np.load(npypath)
    return x.shape[0]


def main():
    parser = argparse.ArgumentParser(description='Generate protocols')

    parser.add_argument('--protocol', default="Protocol_1")

    args = parser.parse_args()
    print(args)

    protoc = args.protocol

    npysfx = "_yuv32.npy"
    rppgsfx = "_yuv8_new.npy"

    sub_dir = "aug_%d_%d" % (windows, stride)

    dstf_dir = os.path.join(os.path.join(proto_root_dir, protoc), sub_dir)

    if not os.path.exists(dstf_dir):
        os.mkdir(dstf_dir)

    # for protocol 1 and 2
    if protoc == "Protocol_1" or protoc == "Protocol_2":
        protocol_dir = os.path.join(proto_root_dir, protoc)
        # print(protocol_dir)
        for itype, clss in enumerate(classes):
            srcf_path = os.path.join(protocol_dir, clss + ".txt")
            dstf_path = os.path.join(dstf_dir, clss + "_32_proto.txt")
            errf_path = os.path.join(dstf_dir, clss + "_32_err.txt")

            npy_base_dir = os.path.join(root_dir, npy_dirs[itype])
            rppg_base_dir = os.path.join(root_dir, rppg_dirs[itype])
            # print(srcf_path)
            # print(dstf_path)
            
            srcf = open(srcf_path, 'r')
            dstf = open(dstf_path, 'w')
            errf = open(errf_path, 'w')
            
            line = srcf.readline()
            while line:
                label_str, videoid = parseOULUProtocol(line.strip('\n'))
                # print("{}, {}".format(label_str, videoid))
                
                if label_str == "+1":
                    label = 1
                else:
                    label = 0

                npy_path = os.path.join(npy_base_dir, videoid + npysfx)
                rppg_path = os.path.join(rppg_base_dir, videoid + rppgsfx)
                
                flag = True
                if not os.path.exists(npy_path):
                    errf.write(npy_path + "\n")
                    flag = False
                
                if label == 1 and not os.path.exists(rppg_path):
                    errf.write(rppg_path + "\n")
                    flag = False
                
                framesnum = get_frames(npy_path)
                if framesnum < windows:
                    errf.write(npy_path + " frames: " + str(framesnum) + "\n")
                    flag = False
                
                st = 0
                ed = windows

                while (ed < framesnum):
                    if flag:
                        dstf.write(npy_path + " " + rppg_path + " " + str(st) + " " + str(ed) + " " + str(label) + "\n")
                        st += stride
                        ed += stride
                
                line = srcf.readline()
                
            srcf.close()
            dstf.close()
            errf.close()

    # for protocol 3 and 4
    if protoc == "Protocol_3" or protoc == "Protocol_4":
        protocol_dir = os.path.join(proto_root_dir, protoc)
        # print(protocol_dir)
        for itype, clss in enumerate(classes):
            for idx in range(1, 7):
                srcf_path = os.path.join(protocol_dir, clss + "_%d.txt" % idx)
                dstf_path = os.path.join(dstf_dir, clss + "_%d_32_proto.txt" % (idx))
                errf_path = os.path.join(dstf_dir, clss + "_%d_32_err.txt" % (idx))

                npy_base_dir = os.path.join(root_dir, npy_dirs[itype])
                rppg_base_dir = os.path.join(root_dir, rppg_dirs[itype])
                # print(srcf_path)
                # print(dstf_path)
                
                srcf = open(srcf_path, 'r')
                dstf = open(dstf_path, 'w')
                errf = open(errf_path, 'w')
                
                line = srcf.readline()
                while line:
                    label_str, videoid = parseOULUProtocol(line.strip('\n'))
                    # print("{}, {}".format(label_str, videoid))
                    
                    if label_str == "+1":
                        label = 1
                    else:
                        label = 0

                    npy_path = os.path.join(npy_base_dir, videoid + npysfx)
                    rppg_path = os.path.join(rppg_base_dir, videoid + rppgsfx)
                    
                    

                    flag = True
                    if not os.path.exists(npy_path):
                        errf.write(npy_path + "\n")
                        flag = False
                    
                    if label == 1 and not os.path.exists(rppg_path):
                        errf.write(rppg_path + "\n")
                        flag = False

                    framesnum = get_frames(npy_path)
                    if framesnum < windows:
                        errf.write(npy_path + " frames: " + str(framesnum) + "\n")
                        flag = False
                    
                    st = 0
                    ed = windows

                    while (ed < framesnum):
                        if flag:
                            dstf.write(npy_path + " " + rppg_path + " " + str(st) + " " + str(ed) + " " + str(label) + "\n")
                            st += stride
                            ed += stride
                    
                    line = srcf.readline()
                    
                srcf.close()
                dstf.close()
                errf.close()

if __name__=="__main__":
    main()