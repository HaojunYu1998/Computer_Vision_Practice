import os
import time
import pickle
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score
from typing import Any, Callable, List, Optional, Tuple

def global_contrast_normalization(img, s, lamda, eps):
    img_avg = np.mean(img)
    img = img - img_avg
    contrast = np.sqrt(lamda + np.mean(img**2))
    rescaled_img = (img - img.min()) / (img.max() - img.min())
    return np.array(rescaled_img, dtype=np.float32)

def ZCA_whitening(imgs):
    flat_imgs = imgs.reshape(imgs.shape[0], -1)
    norm_imgs = flat_imgs / 255.
    norm_imgs = norm_imgs - norm_imgs.mean(axis=0)
    cov = np.cov(norm_imgs, rowvar=False)
    U,S,V = np.linalg.svd(cov)
    epsilon = 0.1
    white_imgs = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(norm_imgs.T).T
    rescaled_white_imgs = white_imgs.reshape(-1, 3, 32,32)
    return rescaled_white_imgs

def transform(img):
    img = (img - img.mean()) / np.std(img)
    # return global_contrast_normalization(img, 1, 10, 0.1)
    return np.array(img, dtype=np.float32)

def target_transform(label):
    return np.array(label, dtype=np.float32)

def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1, padding=1):
    """1x1 convolution with padding"""
	
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(lr, epoch, decay_epochs):
    return lr * 0.2 if (epoch+1) in decay_epochs else lr

def save_model(model, optimizer, epoch, file_name, model_folder) -> None:
    print("==> Saving...")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "depth": model.depth,
        "widen_factor": model.widen_factor,
        "dropout_rate": model.dropout_rate,
        "num_classes": model.num_classes,
    }
    save_file = os.path.join(
        model_folder,
        file_name,
    )
    torch.save(state, save_file)
    del state

def print_info(epoch, num_epochs, loss, acc, epoch_time):
    print(
        "epoch: [{0}/{1}]\t"
        "training loss: {loss.avg:.3f}\t"
        "training accuracy: {acc.avg:.3f}\t"
        "time: {epoch_time.val:.3f}".format(
            epoch, num_epochs, loss=loss, acc=acc, epoch_time=epoch_time,
        )
    )

def plot_history(depth, widen_factor):
    history = np.load("model/WRN_{}_{}/history.npy".format(depth, widen_factor), allow_pickle=True).item()
    plt.figure(figsize=(12,8))
    plt.plot(history["accuracy"])
    plt.plot(history["loss"])
    plt.savefig("model/WRN_{}_{}/history.png".format(depth, widen_factor))
