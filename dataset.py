import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torch
import numpy as np
from numpy import float32
import os
import random
# from options import opt

class AlignedDataset(data.Dataset):
    def __init__(self, file_list="",input_nc = 3,output_nc=1,isTrain = True): #/train or /test

        # input_nc is input image channle (A)
        super(AlignedDataset, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.isTrain = isTrain
        self.data_balance = False
        # self.data_balance = opt.data_balance
        # if self.data_balance:
        #     print("using data_balance")
        self.AB_file_list = file_list
        self.AB_paths = self.get_file_list()

        self.tt = transforms.ToTensor()

        if self.isTrain:
            self.A_transform = self.A_transform_train
        else:
            self.A_transform = self.A_transform_test

    def get_file_list(self):
        "/mnt/hdd.user/datasets/FAS/Oulu-NPU/Train_npy/1_1_01_1_rgb128.npy /mnt/hdd.user/datasets/FAS/Oulu-NPU/Train_rppg/1_1_01_1.npy 1"
        A_path = []
        B_path = []
        label = []
        for x in open(self.AB_file_list):
            A_path.append(x.strip().split(' ')[0])
            B_path.append(x.strip().split(' ')[1])
            label.append(int(x.strip().split(' ')[2]))
            if self.isTrain and int(x.strip().split(' ')[2])==1:
                A_path.append(x.strip().split(' ')[0])
                B_path.append(x.strip().split(' ')[1])
                label.append(int(x.strip().split(' ')[2]))
                if self.data_balance:
                    A_path.append(x.strip().split(' ')[0])
                    B_path.append(x.strip().split(' ')[1])
                    label.append(int(x.strip().split(' ')[2]))

        return (A_path,B_path,label)

    def A_transform_train(self, video):
        # Apply same transform for frames in the video
        # video numpy -> list of PIL image & do transform -> tensor
        
        cnt = 0
        hflip_rand = random.random()
        angle = transforms.RandomRotation.get_params([-15, 15])
        # print(angle)
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.9, 1.1)
        saturation = random.uniform(0.9, 1.1)
        l = []
        for f in video:
            if cnt >= 64:
                break
            pic = self.tt(f)
            # RandomHorizontalFlip
            if hflip_rand > 0.5:
                pic = tf.hflip(pic)
            # RandomRotation
            pic = tf.rotate(pic, angle)
            # ColorJitter
            pic = tf.adjust_brightness(pic, brightness)
            pic = tf.adjust_contrast(pic, contrast)
            pic = tf.adjust_saturation(pic, saturation)
            # Normalize
            # pic = tf.normalize(pic, mean=[0.59416118, 0.51189164, 0.45280306],
                                    #  std=[0.25687563, 0.26251543, 0.26231294])
            l.append(pic)
            cnt += 1
        ret = torch.stack(l, dim=1)
        return ret

    def A_transform_test(self, video):
        # without transform
        l = []
        cnt = 0
        for f in video:
            if cnt >= 64:
                break
            pic = self.tt(f)
            # pic = Image.fromarray(np.uint8(f))
            # # To_Tensor
            # pic_aug = tf.to_tensor(pic)
            # Normalize
            # pic = tf.normalize(pic, mean=[0.59416118, 0.51189164, 0.45280306],
                                    #  std=[0.25687563, 0.26251543, 0.26231294])
            l.append(pic)
            cnt += 1
        ret = torch.stack(l, dim=1)
        return ret

    def __getitem__(self, index):
        A_path = self.AB_paths[0][index]
        B_path = self.AB_paths[1][index]
        label = self.AB_paths[2][index]
        A_yuv128 = np.load(A_path)
        if os.path.exists(B_path):
            B = np.load(B_path).astype(float32)
        else:
            B = np.zeros(A_yuv128.shape[0], dtype=float32)
        A = self.A_transform(A_yuv128)
        B = B[:64]
        B = torch.tensor(B)
        return {'A': A, 'B': B, 'label': label, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths[0])
