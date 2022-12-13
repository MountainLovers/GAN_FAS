import os
import cv2
import threading
import argparse

root_dir = "/mnt/hdd.user/datasets/FAS/Oulu-NPU"
proto_root_dir = "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Protocols"
protocols = ["Protocol_1", "Protocol_2", "Protocol_3", "Protocol_4"]
classes = ["Train", "Dev", "Test"]
npy_dirs = ["Train_npynew", "Dev_npynew", "Test_npynew"]
rppg_dirs = ["Train_rppg", "Dev_rppg", "Test_rppg"]

def parseOULUProtocol(str):
    label = str.split(',')[0]
    name = str.split(',')[1]
    return label, name


def main():
    parser = argparse.ArgumentParser(description='Generate protocols')

    parser.add_argument('--protocol', default="Protocol_1")

    args = parser.parse_args()
    print(args)

    protoc = args.protocol

    npysfx = "_yuv32.npy"
    rppgsfx = "_yuv8_new.npy"
    # for protocol 1 and 2
    if protoc == "Protocol_1" or protoc == "Protocol_2":
        protocol_dir = os.path.join(proto_root_dir, protoc)
        # print(protocol_dir)
        for itype, clss in enumerate(classes):
            srcf_path = os.path.join(protocol_dir, clss + ".txt")
            dstf_path = os.path.join(protocol_dir, clss + "_32_proto.txt")
            errf_path = os.path.join(protocol_dir, clss + "_32_err.txt")

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
                
                if flag:
                    dstf.write(npy_path + " " + rppg_path + " " + str(label) + "\n")
                
                line = srcf.readline()
                
            srcf.close()
            dstf.close()
            errf.close()

if __name__=="__main__":
    main()