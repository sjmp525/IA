import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # pay attention !
import time
import logging
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import resnet

import random
import numpy as np

from tensorboardX import SummaryWriter
import torchvision.utils as vutils

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_1w1a', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet20_1w1a)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 1000')
parser.add_argument('--warm_up_epochs', default=16, type=int, metavar='N',
                    help='warm_up_epochs to run (default: 16')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.10)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum ！！！ (default: 0.9)')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--num_classes', default=10, type=int, metavar='Num_Classes')
parser.add_argument('--seed', default=0, type=int, metavar='Seed', help='setup my seed (default 2023 or 0:random)')

parser.add_argument('--dynamic_range', default=[0.8, 0.0], type=float, metavar='dynamic_range',
                    help='(dong de dou dong!)')

parser.add_argument('--resume', default='none', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_false',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='saveModel-XXX', type=str)
parser.add_argument('--print_freq', '-p', type=int, default=100,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_every', dest='save_every', type=int, default=200,
                    help='Saves checkpoints at every specified number of epochs')

best_prec1 = 0.


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    if args.seed == 0:
        torch.backends.cudnn.deterministic = False  # 不确定为默认卷积算法
        torch.backends.cudnn.benchmark = True  # 模型卷积层算法预先优化打开
        print("\n************ The Model has been Random ************\n")
    else:
        """
        操作:   固定网络的所有随机数，使模型结果可复现
        """
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)  # 禁止hash随机化
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)  # CPU随机种子确定

        torch.cuda.manual_seed(args.seed)  # GPU随机种子确定
        torch.cuda.manual_seed_all(args.seed)  # 所有的GPU设置种子
        torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法
        torch.backends.cudnn.benchmark = False  # 模型卷积层算法预先优化关闭

        print("\n************ The Model has been Seeded ************\n")

    # record_log Record
    if not os.path.exists('record_log'):
        os.mkdir('record_log')
    logging.basicConfig(level=logging.INFO, filename='record_log/' + ''.join(args.arch) + '-XXX.log', format='%(message)s')
    logging.info(args)
    logging.info('Epoch\t''train_loss\t''val_loss\t''train_acc\t''val_acc\t')
    # Tensorboard Record
    SumWriter = SummaryWriter(log_dir='log-XXX')

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # CIFAR10 使用
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/home/kmyh/myData/CIFAR10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4, padding_mode='edge'),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/home/kmyh/myData/CIFAR10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
    warm_up_with_cosine_lr = lambda epoch: (epoch+1) / (args.warm_up_epochs+1) if epoch <= args.warm_up_epochs \
        else 0.5 * (math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this implementation it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    if args.evaluate:
        prec1, val_loss = validate(val_loader, model, criterion)
        print(' *** Prec@1 {:.3f}\t'.format(prec1))
        return

    print(model.module)

    T_min, T_max = 1e-3, 1e1
    def Log_UP(K_min, K_max, epoch):
        Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
        return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / args.epochs * epoch)]).float().cuda()

    print("\n************************ Start ************************\n")
    for epoch in range(args.start_epoch, args.epochs):
        t = Log_UP(T_min, T_max, epoch)
        if (t < 1):
            k = 1. / t
        else:
            k = torch.tensor([1]).float().cuda()

        # ----------------------------------- 当前数据的可反馈百分比，由外部传入 ------------------------------------------- #
        # 传递p，即指示此时的分位
        assert args.dynamic_range[0] > args.dynamic_range[1]
        p = args.dynamic_range[0] - (args.dynamic_range[0] - args.dynamic_range[1]) * (epoch / args.epochs)
        for name, module in model.named_modules():
            if 'myapproxsign' in name:
                module.pp = torch.tensor([p]).float().cuda()
        print('current dynamic_p:', p)
        # ------------------------------------------------------------------------------------------------------------ #

        for i in range(3):
            model.module.layer1[i].conv1.k = k
            model.module.layer1[i].conv2.k = k
            model.module.layer1[i].conv1.t = t
            model.module.layer1[i].conv2.t = t

            model.module.layer2[i].conv1.k = k
            model.module.layer2[i].conv2.k = k
            model.module.layer2[i].conv1.t = t
            model.module.layer2[i].conv2.t = t

            model.module.layer3[i].conv1.k = k
            model.module.layer3[i].conv2.k = k
            model.module.layer3[i].conv1.t = t
            model.module.layer3[i].conv2.t = t

        # train for one epoch
        print('current lr: {:.6e}'.format(optimizer.param_groups[0]['lr']))
        StartEpochTime = time.time()
        train_acc, train_loss = train(train_loader,
                                      model,
                                      criterion,
                                      optimizer,
                                      epoch,
                                      SumWriter,
                                      args)
        lr_scheduler.step()
        print("--- Train One Epoch Time(/s) :", (time.time() - StartEpochTime))

        # evaluate on validation set
        prec1, val_loss = validate(val_loader, model, criterion)
        print(' *** Prec@1 {:.3f}\t'.format(prec1))

        # show
        SumWriter.add_scalar("train_loss", train_loss, epoch)
        SumWriter.add_scalar("train_acc", train_acc, epoch)
        SumWriter.add_scalar("test_loss", val_loss, epoch)
        SumWriter.add_scalar("test_acc", prec1, epoch)
        SumWriter.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # Save Model
        save_checkpoint(epoch, args.save_every, {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.save_dir)

        log_message = str(epoch) + '\t' + '{:.3f}'.format(train_loss) + '\t' + '{:.3f}'.format(
            val_loss) + '\t' + '{:.3f}'.format(train_acc) + '\t' + '{:.3f}'.format(prec1)
        logging.info(log_message)

    print('best_Prec@1 {:.3f}'.format(best_prec1))


def train(train_loader, model, criterion, optimizer, epoch, SumWriter, args):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    inference_time = AverageMeter()
    # switch to train mode
    model.train()
    scaler = GradScaler()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()

        # with autocast():
        # compute output
        start = time.time()
        output = model(input)
        end = time.time()

        inference_time.update(end - start)

        loss = criterion(output, target.detach())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        prec1 = accuracy(output.data, target)
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            print('Epoch：[{0}][{1}/{2}]'
                  'Time：{batch_time.val:.3f}({batch_time.avg:.3f}) '
                  'Data：{data_time.val:.3f}({data_time.avg:.3f}) '
                  'Inference Time: {inference_time.val:.3f}({inference_time.avg:.3f}) '
                  'Loss：{loss.val:.4f}({loss.avg:.4f}) '
                  'Prec@1：{top1.val:.3f}({top1.avg:.3f}) '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time,inference_time=inference_time,loss=losses, top1=top1))


    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)

            loss = criterion(output, target.detach())

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1 = accuracy(output.data, target)
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test：[{0}/{1}] '
                      'Time：{batch_time.val:.3f}({batch_time.avg:.3f}) '
                      'Loss：{loss.val:.4f}({loss.avg:.4f}) '
                      'Prec@1：{top1.val:.3f}({top1.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    return top1.avg, losses.avg


def save_checkpoint(epoch, save_every, state, is_best, save):
    """
    Save the training model
    """
    filename = os.path.join(save, str(epoch + 1) + '-checkpoint.th')
    best_filename = os.path.join(save, 'best-checkpoint.th')
    if (epoch + 1) % save_every == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
