import random
import torch
import numpy as np
import argparse
from warmup_scheduler import GradualWarmupScheduler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="EFFNET_NOCSV_RONDOM_TRANS")
    parser.add_argument('--kernel-type', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=str, default=96)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--root-path', type=str, default="/home/xinsheng/skinImage/")
    parser.add_argument('--use-csv-data', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default="./models/")
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--TTA', type=int, default=5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--nums-worker', type=int, default=4)
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