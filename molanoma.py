import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
from PIL import Image
import cv2

import resnext
from resnext import resnext50


train_path = "./jpeg/train/"
test_path = "./jpeg/test/"

train_sheet = pd.read_csv("./train.csv")
test_sheet = pd.read_csv("./test.csv")

train_sheet['image_path'] = train_path + train_sheet['image_name'] + '.jpg'
test_sheet['image_path'] = test_path + train_sheet['image_name'] + '.jpg'

positive_data = train_sheet[train_sheet['target'] == 1]
negative_data = train_sheet[train_sheet['target'] == 0]

labels = {}
partition = {'train': [], 'test': []}
name_to_path = {key: value for key, value in zip(train_sheet['image_name'].tolist(), train_sheet['image_path'].tolist())}
labels = {key: value for key, value in zip(train_sheet['image_name'].tolist(), train_sheet['target'].tolist())}

partition['train'] = negative_data['image_name'].tolist()
partition['test'] = test_sheet['image_name'].tolist()
len(partition['train']), len(partition['test'])

train_data = partition['train'][:500] + positive_data['image_name'].tolist()[:500]
val_data = partition['train'][500:800] + positive_data['image_name'].tolist()[500:]


class SkinImageDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_names, labels, save=False, transform=None):
        'Initialization'
        self.labels = labels
        self.list_names = list_names
        self.transform = transform
        self.save = save

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_names)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if not self.save:
            image_name = self.list_names[index]
            # print(image_name)
            # print(name_to_path[image_name])
            # Load data and get label
            if transform:
                #   X = cv2.imread(name_to_path[image_name])
                  X = Image.open(name_to_path[image_name])
                #   print(type(X))
                  X = self.transform(X)
            else:
                #   X = cv2.imread(name_to_path[image_name])
                  X = Image.open(name_to_path[image_name])
            # X = X.ToTensor()
            y = self.labels[image_name]

            return X, y
        else:
            pass
            # image_name = self.list_names[index]

            # # Load data and get label
            # if transform:
            #       X = Image.open(name_to_path[image_name])
            #       samlple_X = self.transform(X)
            # else:
            #       samlple_X = Image.open(name_to_path[image_name])
            # # X = X.ToTensor()
            # # y = self.labels[image_name]

            # return samlple_X, image_name 


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])

train_dataset = SkinImageDataset(train_data, labels, save=False, transform=transform)
val_dataset = SkinImageDataset(val_data, labels, save=False, transform=transform)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(output)
        target.resize((64, 1))
        print(target.shape)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args['dry_run']:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # print(data.shape)
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            print(f"output is {output}")
            print(target.shape, output.shape)
            target.resize((64, 1))
            
            test_loss +=  F.binary_cross_entropy_with_logits(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


args = {
    'batch_size': 64,
    "epochs": 5,
    "lr": 0.1,
    "gamma": 0.7,
    "no_cuda": False,
    "seed": 1,
    "log_interval": 10,
    "dry_run": True
}

use_cuda = not args['no_cuda'] and torch.cuda.is_available()
torch.manual_seed(args['seed'])
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'batch_size': args['batch_size']}
if use_cuda:
    kwargs.update({'num_workers': 4,
                    'pin_memory': True,
                    'shuffle': True},
                    )


resnext = resnext50(4, 32)
print(resnext.num_classes)
model = resnext.to(device)
model = torch.nn.DataParallel(model)
optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])

train_dataloader = DataLoader(train_dataset, **kwargs)
print(len(train_dataloader))
val_dataloader = DataLoader(val_dataset, **kwargs)
print(len(val_dataloader))


scheduler = StepLR(optimizer, step_size=1, gamma=args['gamma'])
for epoch in range(1, args['epochs'] + 1):
    train(args, model, device, train_dataloader, optimizer, epoch)
    test(model, device, val_dataloader)
    # save_result(model, device, save_dataset)
    scheduler.step()
