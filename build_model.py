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
    def __init__(self, output_size, type, no_columns, meta_features=None):
        super().__init__()
        self.type = type
        self.output_size = output_size
        self.no_columns = len(meta_features)
        self.meta_features = meta_features
        # Define Feature part (IMAGE)
        if type == 4:
            self.features = EfficientNet.from_pretrained('efficientnet-b4')
        elif type == 2:
            self.features = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            self.features = EfficientNet.from_pretrained('efficientnet-b7')

        # (CSV) not using data from CSV file currently
        self.meta = nn.Sequential(nn.Linear(self.no_columns, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),

                                 nn.Linear(512, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))

        # Define Classification part
        if self.type == 4:
            self.classification = nn.Sequential(nn.Linear(1792 + 128, output_size))
        elif self.type == 2:
            self.classification = nn.Sequential(nn.Linear(1408 + 128, output_size))
        else:
            self.classification = nn.Sequential(nn.Linear(2560 + 128, output_size))

    def forward(self, image, meta_data, prints=False):
        # IMAGE CNN
        image = self.features.extract_features(image)
        if prints: print('Features Image shape:', image.shape)

        if self.type == 4:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1792)
        elif self.type == 2:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408)
        else:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 2560)
        if prints: print('Image Reshaped shape:', image.shape)

        # using meta data from csv file
        meta_data = self.meta(meta_data)

        if prints: print('CSV Data:', meta_data.shape)
        # Concatenate
        image_meta_features = torch.cat((image, meta_data), dim=1)

        # CLASSIF
        out = self.classification(image_meta_features)
        if prints: print('Out shape:', out.shape)

        return out
