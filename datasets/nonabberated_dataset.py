from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset



class ElectronNonAbTrainDataset(Dataset):
    def __init__(self, dataset_dir='/baldig/physicsprojects/electron_microscopy', downsample_rate=1):
        print("Loading Data")
        self.data = np.load(Path(dataset_dir) / 'Training_non-ab.npy', mmap_mode='r')
        print("Data Loaded!")

        max_frame = self.data.shape[2]
        trios = []

        for start in range(max_frame - 2 * downsample_rate):
            # Compute the trio
            trio = np.array([start + i * downsample_rate for i in range(3)])
            
            # If all elements are in the same 181-block, keep it
            if (trio // 181 == trio[0] // 181).all():
                trios.append(trio)

        self.data_idxs = np.array(trios)

    def __len__(self):
        return len(self.data_idxs)
    
    def __getitem__(self, idx):
        idxs = self.data_idxs[idx]
        img0 = self.data[:,:,idxs[0]]
        img1 = self.data[:,:,idxs[1]]
        img2 = self.data[:,:,idxs[2]]

        if img0.shape == (512, 512, 3):

            img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32))
            img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32))
            img2 = torch.from_numpy(img2.transpose((2, 0, 1)).astype(np.float32))

            
        elif img0.shape == (512, 512):

            img0 = torch.from_numpy(np.expand_dims(img0, axis=0).astype(np.float32))
            img1 = torch.from_numpy(np.expand_dims(img1, axis=0).astype(np.float32))
            img2 = torch.from_numpy(np.expand_dims(img2, axis=0).astype(np.float32))

        img0 -= torch.min(img0)
        img0  /= (torch.max(img0) - torch.min(img0))
        img1 -= torch.min(img1)
        img1  /= (torch.max(img1) - torch.min(img1))
        img2 -= torch.min(img2)
        img2  /= (torch.max(img2) - torch.min(img2))
    
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))
            
        return img0, img1, img2, embt


class ElectronNonAbTestDataset(Dataset):
    def __init__(self, dataset_dir='/baldig/physicsprojects/electron_microscopy', downsample_rate=1):

        print("Loading Data")
        self.data = np.load(Path(dataset_dir) / 'Test_non-ab.npy', mmap_mode='r')
        print("Data Loaded!")

        max_frame = self.data.shape[2]
        trios = []

        for start in range(max_frame - 2 * downsample_rate): # range(181 - (2 * downsample_rate)):
            # Compute the trio
            trio = np.array([start + i * downsample_rate for i in range(3)])
            
            # If all elements are in the same 181-block, keep it
            if (trio // 181 == trio[0] // 181).all():
                trios.append(trio)

        self.data_idxs = np.array(trios)
    
    def __len__(self):
        return len(self.data_idxs)
    
    def __getitem__(self, idx):
        idxs = self.data_idxs[idx]
        img0 = self.data[:,:,idxs[0]]
        img1 = self.data[:,:,idxs[1]]
        img2 = self.data[:,:,idxs[2]]

        if img0.shape == (512, 512, 3):

            img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32))
            img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32))
            img2 = torch.from_numpy(img2.transpose((2, 0, 1)).astype(np.float32))

            
        elif img0.shape == (512, 512):
            
            img0 = torch.from_numpy(np.expand_dims(img0, axis=0).astype(np.float32))
            img1 = torch.from_numpy(np.expand_dims(img1, axis=0).astype(np.float32))
            img2 = torch.from_numpy(np.expand_dims(img2, axis=0).astype(np.float32))


        img0 -= torch.min(img0)
        img0  /= (torch.max(img0) - torch.min(img0))
        img1 -= torch.min(img1)
        img1  /= (torch.max(img1) - torch.min(img1))
        img2 -= torch.min(img2)
        img2  /= (torch.max(img2) - torch.min(img2))
    

        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))
            

        return img0, img1, img2, embt
