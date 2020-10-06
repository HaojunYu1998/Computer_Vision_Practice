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

class CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data: Any = []
        self.targets = []

        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list

        for file_name, checksum in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1') # 0-255
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img



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
