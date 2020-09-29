mport os
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
from wide_resnet import WideResNet
from dataset import CIFAR10
from utils import transform, target_transform, AverageMeter, adjust_learning_rate, print_info, save_model

def train_one_epoch(train_loader, model, criterion, optimizer, learning_rate, gpu=1) -> Any:
    n_batch = len(train_loader)
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for dx, batch in enumerate(train_loader):
        img, label = batch
        img = torch.Tensor(img).to(torch.device(gpu))
        label = torch.Tensor(label).view(-1,1).to(torch.device(gpu))
        batch_size = img.shape[0]

        out = model(img)
        loss = criterion(out, label.long().squeeze())
        optimizer.zero_grad()
        loss.backward()
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        optimizer.step()

        preds = out.argmax(dim=1)
        acc = accuracy_score(label.cpu().long().squeeze(), preds.cpu().long().squeeze())
        acc_meter.update(acc, batch_size)
        loss_meter.update(loss.item(), batch_size)

        #torch.cuda.synchronize()
        
    return loss_meter, acc_meter
    
def train(
    depth: int,
    widen_factor: int,
    dropout_rate: float=0.3,
    num_classes: int=10,
    num_epochs: int=200,
    lr: float=0.1,
    decay_epochs: List=[60, 120, 160],
    verbose: bool=True,
    print_freq: int=10,
    save_freq: int=10,
    gpu: int=1,
) -> None:
    torch.manual_seed(2020)
    torch.backends.cudnn.deterministic = True
    np.random.seed(2020)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_folder = os.path.join("model/","WRN_{}_{}/".format(depth, widen_factor))

    train_loader = DataLoader(CIFAR10("./data", True, transform, target_transform), batch_size=128, shuffle=True)
    print("finish loading data!")

    model = WideResNet(depth, widen_factor, dropout_rate, num_classes)
    model = model.to(torch.device(gpu))
    print("finish creating model!")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 200

    epoch_time = AverageMeter()
    print("start training!")

    since = time.time()
    history = {"accuracy": [], "loss": []}
    for epoch in range(1, num_epochs+1):

        lr = adjust_learning_rate(lr, epoch, decay_epochs)
        loss, acc = train_one_epoch(train_loader, model, criterion, optimizer, lr, gpu)
        epoch_time.update(time.time() - since)

        if verbose and epoch % print_freq == 0:
            print_info(epoch, num_epochs, loss, acc, epoch_time)
            history["accuracy"].append(acc.avg)
            history["loss"].append(loss.avg)

        if epoch % save_freq == 0:
            save_model(model, optimizer, epoch, "ckpt_epoch_{epoch}.pth".format(epoch=epoch), model_folder)
    save_model(model, optimizer, num_epochs, "current.pth", model_folder)
    torch.cuda.empty_cache()
    print("finish training!")
    np.save("./model/WRN_{}_{}/history.npy".format(depth, widen_factor), history)

def test(
    depth: int,
    widen_factor: int,
    gpu: int=1,
) -> Any:
    print("Test WRN_{}_{}".format(depth, widen_factor))
    test_loader = DataLoader(CIFAR10("./data", False, transform, target_transform), batch_size=128, shuffle=True)
    checkpoint = torch.load("./model/WRN_{}_{}/current.pth".format(depth, widen_factor), map_location="cpu")
    model = WideResNet(checkpoint["depth"], checkpoint["widen_factor"], checkpoint["dropout_rate"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(torch.device(gpu))
    model.eval()
    acc_meter = AverageMeter()
    for idx, batch in enumerate(test_loader):
        img, label = batch
        img = torch.FloatTensor(img).view(-1,3,32,32).to(torch.device(gpu))
        label = torch.FloatTensor(label).view(-1,1).to(torch.device(gpu))
        batch_size = img.shape[0]
        out = model(img)
        preds = out.argmax(dim=1)
        acc = accuracy_score(label.cpu().long().squeeze(), preds.cpu().long().squeeze())
        acc_meter.update(acc, batch_size)
    print("test accuracy: {}".format(acc_meter.avg))
    np.save("./model/WRN_{}_{}/accuracy.npy".format(depth, widen_factor), np.array([acc_meter.avg]))
    return acc_meter

if __name__ == "__main__":
    train(28, 10, print_freq=1)
                                                                                                                                   125,31        Bo
