import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image,ImageOps,ImageFilter
from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise,
                            Rotate, RandomResizedCrop, Cutout, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2, ToTensor
import cv2
import random
import albumentations
import visdom
from utils import UnNormalize


train_mask_path = "../U-2-Net/test_data/u2net_train_512_results/"
test_mask_path = "../U-2-Net/test_data/u2net_test_512_results/"

viz = visdom.Visdom()
viz2 = visdom.Visdom()
viz3 = visdom.Visdom()
viz4 = visdom.Visdom()


class MelanomaDataset():
    def __init__(self, df: pd.DataFrame, imfolder: str, type, train: bool = True, transforms = None, meta_features = None):
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
        self.type = type
        self.count = 0
        
    def __getitem__(self, index):
        mask_name = self.df.iloc[index]['image_name'] + ".png"
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + ".jpg")
        x = cv2.imread(im_path)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        # meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)

        if self.type == "train":
            mask_path = train_mask_path
        else:
            mask_path = test_mask_path

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.transforms:
            res = self.transforms(image=x)
            # for albumentations transforms
            x = res['image'].astype(np.float32)
        x = albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image=x)['image']
        # after_transform = torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32)
        original_image = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32))

        mask = cv2.imread(mask_path + mask_name)
        original_mask = mask
        random.seed(seed)  # apply this seed to target tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.transforms is not None:
            res = self.transforms(image=mask)
            mask = res['image'].astype(np.float32)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)  # mask should be np.uint8 type or will be an error
        ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        x = cv2.bitwise_and(x, x, mask=mask)

        # for ablumentations
        x = x.transpose(2, 0, 1)
        # data = torch.tensor(x).float()
        self.count += 1
        if self.count <= 2:
            viz.image(original_image)
            viz2.image(mask, env="mask")
            viz3.image(x, env="after mask")
            viz4.image(original_mask, env="original mask")

        if self.meta_features:
            data = (torch.tensor(x, dtype=torch.float32), torch.tensor(self.df.iloc[index][self.meta_features], dtype=torch.float32))
        else:
            data = torch.tensor(x, dtype=torch.float32)

        if self.train:
            # y = self.df.iloc[index]['target']
            # for albumentations transforms
            y = torch.tensor(self.df.iloc[index]['target'], dtype=torch.float32)
            return data, y
        else:
            return data
    
    def __len__(self):
        return len(self.df)


class Microscope():
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Args:
        p (float): probability of applying an augmentation
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),  # image placeholder
                                (img.shape[0] // 2, img.shape[1] // 2),  # center point of circle
                                random.randint(img.shape[0] // 2 - 3, img.shape[0] // 2 + 15),  # radius
                                (0, 0, 0),  # color
                                -1)

            mask = circle - 255
            img = np.multiply(img, mask)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


def get_transforms(type="albumentations"):
    if type == "albumentations":
        train_transforms = albumentations.Compose([
            albumentations.Transpose(p=0.5),
            albumentations.OneOf([
                albumentations.VerticalFlip(p=0.5),
                albumentations.HorizontalFlip(p=0.5),
            ]),
            albumentations.OneOf([
                albumentations.RandomBrightness(limit=0.2, p=0.75),
                albumentations.RandomContrast(limit=0.2, p=0.75),
            ]),
            albumentations.OneOf([
                albumentations.MotionBlur(blur_limit=5),
                albumentations.MedianBlur(blur_limit=5),
                albumentations.GaussianBlur(blur_limit=5),
                albumentations.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            albumentations.OneOf([
                albumentations.OpticalDistortion(distort_limit=1.0),
                albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                albumentations.ElasticTransform(alpha=3),
            ], p=0.7),

            # albumentations.OneOf([
            #     albumentations.CLAHE(clip_limit=4.0, p=0.7),
            #     albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            #     albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0,
            #                                     p=0.85),
            # ]),
            albumentations.Resize(256, 256),
            # albumentations.Cutout(max_h_size=int(256 * 0.375), max_w_size=int(256 * 0.375), num_holes=1, p=0.7),
            # albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transforms = albumentations.Compose([
            albumentations.Resize(256, 256),
            # albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            # AdvancedHairAugmentation(hairs_folder='/kaggle/input/melanoma-hairs'),
            transforms.RandomResizedCrop(size=256, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Microscope(p=0.5),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        test_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    return train_transforms, test_transforms

