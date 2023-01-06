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
from loguru import logger
from options import opt

class AlignedDataset(data.Dataset):
    def __init__(self, file_list="",input_nc = 3,output_nc=1,isTrain = True, scale=20): #/train or /test

        # input_nc is input image channle (A)
        super(AlignedDataset, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.isTrain = isTrain
        self.data_balance = opt.data_balance
        self.scale = scale
        # if self.data_balance:
        #     print("using data_balance")
        self.AB_file_list = file_list
        self.AB_paths, self.tnum, self.fnum = self.get_file_list()

        self.tt = transforms.ToTensor()

        if self.isTrain:
            self.A_transform = self.A_transform_train
        else:
            self.A_transform = self.A_transform_test

    def get_file_list(self):
        A_path = []
        B_path = []
        ST = []
        ED = []
        label = []
        tnum = 0
        fnum = 0
        for x in open(self.AB_file_list):
            ap, bp, st, ed, lb = x.strip().split(' ')
            A_path.append(ap)
            B_path.append(bp)
            ST.append(int(st))
            ED.append(int(ed))
            label.append(int(lb))
            if int(lb) == 1:
                tnum += 1
            else:
                fnum += 1
            if self.isTrain and int(lb)==1:
                A_path.append(ap)
                B_path.append(bp)
                ST.append(int(st))
                ED.append(int(ed))
                label.append(int(lb))
                tnum += 1
                if self.data_balance:
                    A_path.append(ap)
                    B_path.append(bp)
                    ST.append(int(st))
                    ED.append(int(ed))
                    label.append(int(lb))
                    tnum += 1
        self.tnum = tnum
        self.fnum = fnum
        return (A_path,B_path,ST,ED,label), tnum, fnum

    def A_transform_train(self, video):
        # Apply same transform for frames in the video
        # video numpy -> list of PIL image & do transform -> tensor
        
        hflip_rand = random.random()
        angle = transforms.RandomRotation.get_params([-15, 15])
        # print(angle)
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.9, 1.1)
        saturation = random.uniform(0.9, 1.1)
        l = []
        for f in video:
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
        ret = torch.stack(l, dim=1)
        return ret

    def A_transform_test(self, video):
        # without transform
        l = []
        for f in video:
            pic = self.tt(f)
            # pic = Image.fromarray(np.uint8(f))
            # # To_Tensor
            # pic_aug = tf.to_tensor(pic)
            # Normalize
            # pic = tf.normalize(pic, mean=[0.59416118, 0.51189164, 0.45280306],
                                    #  std=[0.25687563, 0.26251543, 0.26231294])
            l.append(pic)
        ret = torch.stack(l, dim=1)
        return ret
    
    def scale_rppg(self, sig, scale):
        ret = np.zeros_like(sig)
        for i, v in enumerate(sig):
            ret[i] = v * scale
        return ret


    def __getitem__(self, index):
        A_path = self.AB_paths[0][index]
        B_path = self.AB_paths[1][index]
        st = self.AB_paths[2][index]
        ed = self.AB_paths[3][index]
        label = self.AB_paths[4][index]
        # print("dataset {} np load A_yuv32 start".format(index))
        A_yuv32 = np.load(A_path)
        A = A_yuv32[st:ed, :, :, :]
        # print("dataset {} np load A_yuv32 ok".format(index))
        if os.path.exists(B_path):
            B = np.load(B_path).astype(float32)
        else:
            # B = np.zeros(A_yuv32.shape[0], dtype=float32)
            B = np.random.normal(0,0.2,size=A_yuv32.shape[0])
        # print("dataset {} A_transform start".format(index))
        A = self.A_transform(A)
        # print("dataset {} A_transform ok".format(index))
        # print("dataset {} B start".format(index))
        B = B[st:ed].astype(np.float32)
        B = self.scale_rppg(B, self.scale)
        B = torch.tensor(B)
        label = torch.tensor(label)
        # print("dataset {} B ok".format(index))
        return {'A': A, 'B': B, 'label': label, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths[0])

    def get_item_num(self):
        return self.tnum, self.fnum