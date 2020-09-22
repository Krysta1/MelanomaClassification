import torch
import torch.nn as nn
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class ResNet50(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        # Define Feature part (IMAGE)
        self.features = resnet50(pretrained=True)  # 1000 neurons out

        # Define Classification part
        self.classification = nn.Linear(1000, self.output_size)

    def forward(self, image, prints=False):
        # Image CNN
        image = self.features(image)
        if prints: print('Features Image shape:', image.shape)
        # CLASSIFICATION
        out = self.classification(image)
        if prints: print('Out shape:', out.shape)
        return out


class EfficientNetwork(nn.Module):
    def __init__(self, output_size, type, meta_features=None):
        super().__init__()
        self.type = type
        self.output_size = output_size
        self.meta_features = meta_features

        # Define Feature part (IMAGE)
        if type == 4:
            self.features = EfficientNet.from_pretrained('efficientnet-b4')
            # self.features = EfficientNet.from_name('efficientnet-b4')
            # self.features.to(device=torch.device('cuda'))
        elif type == 2:
            self.features = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            self.features = EfficientNet.from_pretrained('efficientnet-b7')
        # print(self.features)
        # self.features._swish = nn.ReLU()
        # for layer in self.features.children():
        #     print("------------")
        #     print(layer)
        # print(self.features.children())
        # (CSV) not using data from CSV file currently
        if self.meta_features:
            self.meta = nn.Sequential(nn.Linear(len(self.meta_features), 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2),

                                     nn.Linear(512, 128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2))

        # Define Classification part
        if self.type == 4:
            if self.meta_features:
                self.classification = nn.Sequential(nn.Linear(1792 + 128, output_size))
            else:
                self.classification = nn.Sequential(nn.Linear(1792, output_size))
        elif self.type == 2:
            if self.meta_features:
                self.classification = nn.Sequential(nn.Linear(1408 + 128, output_size))
            else:
                self.classification = nn.Sequential(nn.Linear(1408, output_size))
        else:
            if self.meta_features:
                self.classification = nn.Sequential(nn.Linear(2560 + 128, output_size))
            else:
                self.classification = nn.Sequential(nn.Linear(2560, output_size))

    def forward(self, x, prints=False):
        # IMAGE CNN
        if self.meta_features:
            image, meta_data = x[0], x[1]
        else:
            image = x
        # image.to(device=torch.device("cuda"))
        image = self.features.extract_features(image)
        # image = self.frelu(image)
        if prints: print('Features Image shape:', image.shape)

        if self.type == 4:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1792)
        elif self.type == 2:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408)
        else:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 2560)
        if prints: print('Image Reshaped shape:', image.shape)

        # using meta data from csv file
        if self.meta_features:
            meta_data = self.meta(meta_data)

            if prints: print('Meta Data:', meta_data.shape)
            # Concatenate
            image_meta_features = torch.cat((image, meta_data), dim=1)
            out = self.classification(image_meta_features)
        else:
            out = self.classification(image)

        # CLASSIF

        if prints: print('Out shape:', out.shape)

        return out


class EffNet(nn.Module):
    pass


# class FReLU(nn.Module):
#     r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
#     """
#     def __init__(self):
#         super().__init__()
#         pass
#
#     def forward(self, x):
#         # print(x.shape)
#         in_channels = x.shape[1]
#         # print(in_channels)
#
#         x1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels).to(device=torch.device("cuda"))(x)
#         x1 = nn.BatchNorm2d(in_channels).to(device=torch.device("cuda"))(x1)
#         x = torch.max(x, x1)
#         return x