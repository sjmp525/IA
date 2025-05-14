from torchvision import datasets, transforms
import torch
from dataset.caltech import Caltech101
from utils import ext_transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from dataset.tiny_imagenet import TinyImageNet
import torch.distributed as dist
import math
import torchvision
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import cv2

# def collate_fn(batch):
#     return mixupcutmix(*default_collate(batch))
# mixup_transforms = [
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.RandomResizedCrop(224),
#     # 你还可以根据需要添加更多的变换
# ]
# mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

class BLUR(nn.Module):
    def __init__(self,Degree : int,methods = 'Gauess', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.methods = methods
        self.Degree = Degree
    
    def motion_blur(self,image : torch.tensor, degree : int = 12, angle : int = 45):
        '''
            image.shape = (c,h,w)
        '''
        image = image.permute(1,2,0)
        image = np.array(image)
        
    
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred)
        
        return torch.tensor(blurred).permute(2,0,1)
    
    def box_blur(self,image : torch.tensor,Degree : int):
        '''
            image.shape = (c,h,w)
        '''
        device = image.device
        image = image.permute(1,2,0)
        image = np.array(image)
        
        blurred = cv2.boxFilter(image,-1,(Degree,Degree),normalize = False)
        return torch.tensor(blurred).permute(2,0,1)
    
    def mean_blur(self,image : torch.tensor,Degree : int):
        '''
            image.shape = (c,h,w)
        '''
        image = image.permute(1,2,0)
        image = np.array(image)
        blurred = cv2.blur(image,(Degree,Degree))
        return torch.tensor(blurred).permute(2,0,1)
    
    def median_blur(self,image : torch.tensor,Degree : int):
        '''
            image.shape = (c,h,w)
        '''
        image = image.permute(1,2,0)
        image = np.array(image)
        blurred = cv2.medianBlur(image,Degree)
        return torch.tensor(blurred).permute(2,0,1)
    
    def bilateral_blur(self,image : torch.tensor,Degree : int = 25):
        '''
            image.shape = (c,h,w)
        '''
        image = image.permute(1,2,0)
        image = np.array(image)
        
        blurred = cv2.bilateralFilter(image,Degree,1,1)
        return torch.tensor(blurred).permute(2,0,1)
    
 
        
    def forward(self,x : torch.tensor):
        '''
            x.shape = (bs,c,h,w)
        '''
        if self.methods == 'Motion':
            if len(x.shape) == 4:
                for bs in range(x.shape[0]):
                    x[bs,...] = self.motion_blur(x[bs,...],self.Degree)
            else:
                x = self.motion_blur(x,self.Degree)
        elif self.methods == 'Gauess':
            trans = transforms.GaussianBlur(self.Degree,2)
            x = trans(x)
        elif self.methods == 'Box':
            if len(x.shape) == 4:
                for bs in range(x.shape[0]):
                    x[bs,...] = self.box_blur(x[bs,...],self.Degree)
            else:
                x = self.box_blur(x,self.Degree)
        elif self.methods == 'Mean':
            if len(x.shape) == 4:
                for bs in range(x.shape[0]):
                    x[bs,...] = self.mean_blur(x[bs,...],self.Degree)
            else:
                x = self.mean_blur(x,self.Degree)
        elif self.methods == 'Median':
            if len(x.shape) == 4:
                for bs in range(x.shape[0]):
                    x[bs,...] = self.median_blurr(x[bs,...],self.Degree)
            else:
                x = self.median_blur(x,self.Degree)
        elif self.methods == 'Bilateral':
            if len(x.shape) == 4:
                for bs in range(x.shape[0]):
                    x[bs,...] = self.bilateral_blur(x[bs,...],self.Degree)
            else:
                x = self.bilateral_blur(x,self.Degree)
        elif self.methods == 'None':
            pass
        return x

class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def load_data(traindir, valdir, args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    print("Loading training data")
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = TinyImageNet('./data', split='train', download=True, transform=train_transform)

    print("Loading validation data")
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset_test = TinyImageNet('./data', split='val', download=False, transform=val_transform)


    print("Creating data loaders")
    # if args.distributed:
    #     if hasattr(args, "ra_sampler") and args.ra_sampler:
    #         train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
    #     else:
    #         train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    # else:
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

class SignLanguageMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = int(self.data.iloc[idx, 0])  # 标签在第0列
        img = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)  # 图像数据从第1列开始
        img = Image.fromarray(img, mode='L')  # 将numpy数组转换为PIL图像
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloader(args):
    if args.dataset.lower() == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

    elif args.dataset.lower() == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                #  BLUR(int(3),'Box'),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset.lower() == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_root, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                              ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_root, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                              ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset.lower() == 'caltech101':
        train_loader = torch.utils.data.DataLoader(
            Caltech101(args.data_root, train=True, download=args.download,
                       transform=transforms.Compose([
                           transforms.Resize(128),
                           transforms.RandomCrop(128),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            Caltech101(args.data_root, train=False, download=args.download,
                       transform=transforms.Compose([
                           transforms.Resize(128),
                           transforms.CenterCrop(128),
                           transforms.ToTensor(),
                        #    BLUR(int(5),'mean'),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
            batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    elif args.dataset.lower() == 'tiny-imagenet':
            train_dir = os.path.join(args.data_root, "train")
            val_dir = os.path.join(args.data_root, "val")
            dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

            train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
    )
            test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.test_batch_size, sampler=test_sampler, pin_memory=True)
            
    elif args.dataset.lower() == 'sign-language-mnist':
        data_transform = {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            "val": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        }
        train_dataset = SignLanguageMNISTDataset(csv_file=os.path.join(args.data_root, "sign_mnist_train.csv"),
                                                 transform=data_transform["train"])
        val_dataset = SignLanguageMNISTDataset(csv_file=os.path.join(args.data_root, "sign_mnist_test.csv"),
                                               transform=data_transform["val"])
        nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw)

    return train_loader, test_loader
