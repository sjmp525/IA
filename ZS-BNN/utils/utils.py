
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop, Grayscale, RandomHorizontalFlip, RandomRotation
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
import random
import cv2


class ScaleInvariantLoss(nn.Module):
    """This criterion is used in depth prediction task.
    **Parameters:**
        - **la** (int, optional): Default value is 0.5. No need to change.
        - **ignore_index** (int, optional): Value to ignore.
    **Shape:**
        - **inputs**: $(N, H, W)$.
        - **targets**: $(N, H, W)$.
        - **output**: scalar.
    """
    def __init__(self, la=0.5, ignore_index=0):
        super(ScaleInvariantLoss, self).__init__()
        self.la = la
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        size = inputs.size()
        if len(size) > 2:
            inputs = inputs.view(size[0], -1)
            targets = targets.view(size[0], -1)
        
        inv_mask = targets.eq(self.ignore_index)
        nums = (1-inv_mask.float()).sum(1)

        log_d = torch.log(inputs) - torch.log(targets)
        log_d[inv_mask] = 0

        loss = torch.div(torch.pow(log_d, 2).sum(1), nums) - \
            self.la * torch.pow(torch.div(log_d.sum(1), nums), 2)

        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def focal_loss(inputs, targets, alpha=1, gamma=0, size_average=True, ignore_index=255):
    ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    if size_average:
        return focal_loss.mean()
    else:
        return focal_loss.sum()

def soft_cross_entropy(logits, target, T=1.0, size_average=True, target_is_prob=False):
    """ Cross Entropy for soft targets
    
    **Parameters:**
        - **logits** (Tensor): logits score (e.g. outputs of fc layer)
        - **targets** (Tensor): logits of soft targets
        - **T** (float): temperature　of distill
        - **size_average**: average the outputs
        - **target_is_prob**: set True if target is already a probability.
    """
    if target_is_prob:
        p_target = target
    else:
        p_target = F.softmax(target/T, dim=1)
    
    logp_pred = F.log_softmax(logits/T, dim=1)
    # F.kl_div(logp_pred, p_target, reduction='batchmean')*T*T
    ce = torch.sum(-p_target * logp_pred, dim=1)
    if size_average:
        return ce.mean() * T * T
    else:
        return ce * T * T

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if len(x.shape)!=2:
        x = x.view(x.shape[0], -1)
    
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return dist

def kldiv(logits, targets, reduction='batchmean'):
    p = F.log_softmax(logits, dim=1)
    q = F.softmax(targets, dim=1)
    return F.kl_div(p, q, reduction=reduction)

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class AddBlur(object):
    def __init__(self, kernel=3, p=1):
        self.kernel = kernel
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            img = cv2.blur(img, (self.kernel, self.kernel))
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class Custom_Dataset(Dataset):
    def __init__(self, root, transform, csv_path):
        super().__init__()
        self.root = root
        self.transform = transform
        self.csv = csv_path
        df = pd.read_csv(self.csv)
        self.info = df

    def __getitem__(self, index):
        patience_info = self.info.iloc[index]
        file_name = patience_info['name']
        file_path = glob(self.root+'/'+file_name)[0]
        file_name = file_name.split('.')[0]
        label = patience_info['label']
        img = Image.open(file_path)
        if self.transform is not None:
            img = self.transform(img)

        return {'imgs': img, 'labels': label, 'names': file_name}

    def __len__(self):
        return len(self.info)

def get_dataset(imgpath, csvpath, img_size, mode='train', keyword=None):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.CenterCrop((img_size, img_size)),
        AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        AddBlur(kernel=3, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        # transforms.RandomRotation((-20, 20)),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])

    if mode =='train':
            transform = train_transform
    elif mode == 'test':
        transform = test_transform

    dataset = Custom_Dataset(imgpath, transform, csvpath)

    return dataset

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)
    return conf_matrix

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)
    return conf_matrix