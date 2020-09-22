import random
import torch
import numpy as np
import argparse
from warmup_scheduler import GradualWarmupScheduler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="TEST")
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--root-path', type=str, default="/home/xinsheng/skinImage/")
    parser.add_argument('--model_path', type=str, default="./models")
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--TTA', type=int, default=5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--use-meta-features', type=bool, default=False)
    parser.add_argument('--init', type=bool, default=False)
    parser.add_argument('--nums-worker', type=int, default=4)
    parser.add_argument('--arch', type=int, default=4)
    parser.add_argument('--train-path', type=str, default="/home/xinsheng/PycharmProjects/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/data/jpeg-melanoma-512x521/train/")
    parser.add_argument('--test-path', type=str, default="/home/xinsheng/PycharmProjects/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/data/jpeg-melanoma-512x521/test/")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--use-diagnosis', type=bool, default=False)
    parser.add_argument('--pos-weight', type=bool, default=False)
    args, _ = parser.parse_known_args()
    return args

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


def ensemble_in_model(version):
    model_path = f"./models{version}/"
    csv_files = [file for file in os.listdir(model_path) if ".csv" in file]
    data = pd.read_csv(model_path + csv_files[0])
    for i in range(1, len(csv_files)):
        data['target'] += pd.read_csv(model_path + csv_files[i])['target']

    data['target'] /= len(csv_files)
    data.to_csv(f"{model_path}{version}_FINAL.csv", index=False)
    print(f"{version}_FINAL.csv saved")


def cal_oof_auc(target, pred):
    return roc_auc_score(target, pred)


if __name__ == "__main__":
    target = pd.read_csv("./data/train-extra-jpeg-256.csv")['target']
    pred = pd.read_csv("./modelsEffnetB7METAPOSW10/oof_EffnetB7METAPOSW10.csv")
    print(cal_oof_auc(target, pred))
