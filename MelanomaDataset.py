import os
import numpy as np
import pandas as pd
import args_config as args
from torchvision import transforms
from PIL import Image,ImageOps,ImageFilter
from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise,
                            Rotate, RandomResizedCrop, Cutout, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2, ToTensor
import cv2


class MelanomaDataset():
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms = None, meta_features = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age
            
        """
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features
        
    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + ".jpg")
        x = cv2.imread(im_path)
        # meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)

        if self.transforms:
            x = self.transforms(x)
            
        if self.train:
            y = self.df.iloc[index]['target']
            return x, y
        else:
            return x
    
    def __len__(self):
        return len(self.df)


# define transforms for datasets.
def get_transforms(horizontal=0.5, vertical=0.5):
    train_transforms = Compose([RandomResizedCrop(height=224, width=224, scale=(0.4, 1.0)),
             ShiftScaleRotate(rotate_limit=90, scale_limit=[0.8, 1.2]),
             HorizontalFlip(p=horizontal),
             VerticalFlip(p=vertical),
             HueSaturationValue(sat_shift_limit=[0.7, 1.3],
                                hue_shift_limit=[-0.1, 0.1]),
             RandomBrightnessContrast(brightness_limit=[0.7, 1.3],
                                      contrast_limit=[0.7, 1.3]),
             Normalize(),
             ToTensorV2()])

    validation_transforms = Compose([Normalize(),
                                      ToTensorV2()]) 

    transform = {'train': train_transforms, "validation": validation_transforms}
    return transform

