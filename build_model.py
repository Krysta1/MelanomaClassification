import torch
import torch.nn as nn
from torchvision.models import resnet50


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


def res_net_50():
    pass





