import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class ImageDataset(Dataset):

    def __init__(self, data, transform):
        

        self.length = data.length
        self.img_path = data.name_resized
        self.label = data.label
        self.bbox = data.bbox_resized

        self.transform = transform

    def __len__(self):

        return len(self.length)
    
    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = read_image(img_path).float() /255
        img = self.transform(img)
        length = self.length[idx]

        if length <= 5:
            label = np.full((5,), 10 ,dtype=np.int64)
            label[:length] = self.label[idx]

            bbox = np.zeros((5,4) ,dtype=np.float32)
            bbox[:length] = self.bbox[idx]

        if length > 5:
            length = 6 # 6 represents "more than 5"
            label = np.array(self.label[idx][:5], dtype=np.int64)
            bbox = np.array(self.bbox[idx][:5], dtype=np.float32)

        bbox[:, [0, 2]] = bbox[:, [0, 2]] / (96)
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / (64)

        return img, length, label, bbox #normalize bbox

        