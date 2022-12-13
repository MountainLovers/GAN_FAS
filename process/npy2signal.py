import argparse
from networks import N3DED8
from pytorch_datasets import SubjectIndependentTestDataset
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import threading

"""""""""
START ARGPARSE
"""""""""
device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")

## MODEL
RTRPPG = N3DED8()
RTRPPG.to(device)
# checkpoint = torch.load('weights.pth.tar',map_location=torch.device('cpu'))
checkpoint = torch.load('weights.pth.tar',map_location=device)
RTRPPG.load_state_dict(checkpoint['model_state_dict'])


class MySubjectIndependentTestDataset(torch.utils.data.Dataset):
    """This dataset takes the path where the subject folder is located. After loading the data,
    it divides it through a sliding window of length=128, step=1.
    """
 
    def __init__(self, load_path, videoid):
        """
        Args:
            load_path(str): Path where the data is located
            window(int): sliding window length
            step(int): sliding window step
            img_size(int): Squared image size
        """
        
        self.load_path = load_path
        self.name = videoid
        self.window = 128
        self.step = 1
        self.img_size = 8
        # print(os.path.join(self.load_path,self.name+'.npy'))
        self.frames = np.load(os.path.join(self.load_path,self.name+'.npy'))
        self.Frames = np.zeros((self.frames.shape[0],self.img_size,self.img_size,3),dtype=np.float32)    
        
        for i in range(0,self.frames.shape[0]):
            frame = np.array(self.frames[i,:,:,:].copy(),dtype=np.float32)/255.0            
            self.Frames[i,:,:,:] = frame

        # Get windows index
        self.windows = self.getWindows()
            
    # RETURN NUMBER OF WINDOWS
    def __len__(self):
            return len(self.windows)

    # RETURN THE [i] FILE
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Take only current window from big tensor
        frames = self.take_frames_crt_window(self.windows[idx])
        sample = {'x':frames, 'f': self.frames}
        return sample    

    # FUNCTION TO GET THE WINDOWS INDEX
    def getWindows(self):
        windows = []
        for i in range(0,np.size(self.Frames,0)-self.window+1,self.step):
            windows.append((i,i+self.window))
        return windows

    # FUNCTION TO TAKE ONLY THE FRAMES OF THE CURRENT WINDOW AND RETURN A TENSOR OF IT
    def take_frames_crt_window(self,idx):
        frames = torch.zeros((3,self.window,self.img_size,self.img_size)) # list with all frames {3,T,128,128}
        
        # Load all frames in current window
        for j,i in enumerate(range(idx[0],idx[1])):
            frame = self.Frames[i,:,:,:]
            frame = torch.tensor(frame, dtype=torch.float32).permute(2,0,1)#In pythorch channels must be in position 0, cv2 has channels in 2
            frames[:,j,:,:] = frame 
        return frames


def normalize(data):    
    from sklearn.preprocessing import MinMaxScaler
    x = np.asarray(data)
    x = x.reshape(len(x), 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(x)
    scaled_x = scaler.transform(x)
    return scaled_x

def main_exp(RTRPPG, basedir, dstdir, videoid):
    device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
    
    ## DATA MANAGER
    dataset = MySubjectIndependentTestDataset(basedir, videoid) 
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False)

  
    window = dataset.window # Slinding window length
    rPPG = []
    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            sample['x'] = sample['x'].to(device)
            out = RTRPPG(sample['x'])
            out = out-torch.mean(out,keepdim=True,dim=1)/torch.std(out,keepdim=True,dim=1)
            rPPG.append(out.to('cpu').detach().numpy())

    rPPG = np.vstack(rPPG)
    
    np.save(os.path.join(dstdir, videoid + "_pred"), rPPG)
    np.savetxt(os.path.join(dstdir, videoid + "_pred.txt"), rPPG)
    
    ## OVERLAP-ADD PROCESS                
    y_hat = np.zeros(window+len(rPPG)-1)
    for i in range(len(rPPG)):
        y_hat[i:i+window] = y_hat[i:i+window]+rPPG[i]
    
    y_hat = np.squeeze(normalize(y_hat))
    
    np.save(os.path.join(dstdir, videoid), y_hat)
    np.savetxt(os.path.join(dstdir, videoid + ".txt"), y_hat)
    
    # PLOT
    fig, ax = plt.subplots()
    plt.plot(y_hat)
    plt.ylabel("Amplitude")
    plt.xlabel("Time [s]")
    plt.legend(['rPPG'])     
    fig.savefig(os.path.join(dstdir, videoid + ".png"), format='png', dpi=1200)
    plt.close()


all_train_npy = get_all_files(TRAIN_NPY_DIR, ".npy")
all_dev_npy = get_all_files(DEV_NPY_DIR, ".npy")
all_test_npy = get_all_files(TEST_NPY_DIR, ".npy")
all_npy = [all_train_npy, all_dev_npy, all_test_npy]


npy_dirs = [TRAIN_NPY_DIR, DEV_NPY_DIR, TEST_NPY_DIR]
rppg_dirs = [TRAIN_RPPG_DIR, DEV_RPPG_DIR, TEST_RPPG_DIR]

for itype, npypaths in enumerate(all_npy):
    npy_dir = npy_dirs[itype]
    rppg_dir = rppg_dirs[itype]
    
    cnt = 0
    for npypath in npypaths:
        basename = os.path.basename(npypath)
        basedir = os.path.dirname(npypath)
        videoid, extname = os.path.splitext(basename)
        
        flag = True
        suffixs = ["_pred.txt", "_pred.npy", ".txt", ".npy", ".png"]
        for sfx in suffixs:
            chkpath = os.path.join(rppg_dir, videoid + sfx)
            if not os.path.exists(chkpath):
                flag = False
                break
        if flag == False:
            main_exp(RTRPPG, npy_dir, rppg_dir, videoid)
            cnt += 1
            print("type: {}, {}/{}, {} Done".format(itype, cnt, len(npypaths), videoid))
        else:
            cnt += 1
            print("type: {}, {}/{}, {} Already Done".format(itype, cnt, len(npypaths), videoid))