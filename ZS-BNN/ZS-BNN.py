from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import network
from utils.visualizer import VisdomPlotter
from utils.misc import pack_images, denormalize
from dataloader import get_dataloader
import os, random
import numpy as np
from utils.DF_ABNNlib import DA, BinarizeConv2d_BiPer, DeepInversionFeatureHook
from utils.DF_ABNNlib import Hist_Show, Hist_Show_KDE
from network.vgg import vgg_small_1w1a
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torchvision
from sklearn.decomposition import PCA
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.DF_ABNNlib import Log_UP
# vp = VisdomPlotter('8097', env='DFAD-cifar')

def pca_reduce_features(t5, output_dim=64):
    """
    使用 PCA 将特征从 512 维降到 64 维
    :param t5: 输入特征，形状为 (N, 512)
    :param output_dim: 输出维度，默认为 64
    :return: 降维后的特征，形状为 (N, output_dim)
    """
    # 将 t5 转换为 numpy 数组
    t5_np = t5.detach().cpu().numpy()

    # 使用 PCA 进行降维
    pca = PCA(n_components=output_dim)
    t5_reduced_np = pca.fit_transform(t5_np)

    # 将结果转换回 torch 张量
    t5_reduced = torch.tensor(t5_reduced_np, dtype=t5.dtype, device=t5.device)
    return t5_reduced

def train(args, teacher, student, generator, device, optimizer, epoch, bn_hook=None, t_hooks=None, tbn_stats=None, s_hooks=None, sbn_stats=None, SumWriter=None):
    teacher.eval()
    student.train()
    generator.train()
    optimizer_S, optimizer_G = optimizer
    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
    CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    # epoch_itrs代表了生成n个batch的数据，对应于数据驱动训练的for x, y in range(train_loader)
    for i in range(args.epoch_itrs):
        # 固定生成器不动，学生网络迭代5次
        for k in range (5):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
            optimizer_S.zero_grad()
            fake = generator(z).detach()

            feat_s, logit_s = student(fake) # 学生网络的特征与输出预测
            feat_t, logit_t = teacher(fake) # 教师网络的特征与输出预测
            loss_1kd = F.l1_loss(logit_s, logit_t.detach())
            loss_S = loss_1kd 
            loss_S.backward()
            optimizer_S.step()
        z = torch.randn((args.batch_size, args.nz, 1, 1)).to(device)
        optimizer_G.zero_grad()
        generator.train()
        fake = generator(z)
        feat_s, logit_s = student(fake)
        feat_t, logit_t = teacher(fake)
        loss_1kd = F.l1_loss(logit_s, logit_t)

        # BN约束项
        P = nn.functional.softmax(logit_s / 3, dim=1)
        Q = nn.functional.softmax(logit_t / 3, dim=1)
        M = 0.5 * (P + Q)
        P = torch.clamp(P, 0.01, 0.99)
        Q = torch.clamp(Q, 0.01, 0.99)
        M = torch.clamp(M, 0.01, 0.99)
        eps = 0.0
        loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
        loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

        # 特征作余弦距离约束
        # P = pca_reduce_features(feat_t, output_dim=64)
        P = feat_t
        Q = feat_s
        # Q = pca_reduce_features(feat_s, output_dim=512)
        M = 0.5 * (P + Q)
        cosine_distance = 1.0 - CosineSimilarity(Q, M)
        loss_bn1 = torch.clamp(0.3 - cosine_distance, min=0)
        loss_bn1 = loss_bn1.mean()
        loss_bn = args.α * sum([mod.r_feature for mod in bn_hook])
        loss_bn += args.β * loss_bn1  # default=0.5

        loss_G = -loss_1kd + loss_bn
        loss_G.backward()
        optimizer_G.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f} loss_bn: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item(), loss_bn.item()))

            # vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            # vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())

def test(args, student, generator, device, test_loader, epoch=0):
    student.eval()
    generator.eval()

    # 初始化指标
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0

    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        
        # 准确率计算
        with torch.no_grad():
            kong, output = student(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred_top1 = output.argmax(dim=1, keepdim=True)
            correct_top1 += pred_top1.eq(target.view_as(pred_top1)).sum().item()
            _, pred_top5 = output.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += pred_top5.eq(target.view(-1, 1).expand_as(pred_top5)).sum().item()


    # 计算各项指标
    test_loss /= len(test_loader.dataset)
    top1_acc = 100. * correct_top1 / len(test_loader.dataset)
    top5_acc = 100. * correct_top5 / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Top-1 Accuracy: {:.4f}%, Top-5 Accuracy: {:.4f}%'.format(
        test_loss, top1_acc, top5_acc))
    
    return top1_acc, top5_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')

    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--epoch_itrs', type=int, default=50) 
    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=0.001,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--data_root', type=str, default='/home/ghw/data/CIFAR10') #  /home/ghw/data/101_ObjectCategories_split

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100','caltech101', 'tiny-imagenet'],
                         help='dataset name (default: cifar10)')
    parser.add_argument('--model', type=str, default='resnet18_1w1a', choices=['resnet18_1w1a, resnet20_1w1a, vgg_small_1w1a', 'resnet34_t', 'resnet18_1w1a_ti', 'resnet34_1w1a_ti'],
                        help='model name (default: resnet18_1w1a)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt') 
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--test-only', action='store_true', default=True)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--step_size', type=int, default=100, metavar='S')
    parser.add_argument('--scheduler', action='store_true', default=True)
    parser.add_argument('--progressive', dest='progressive', action='store_true',
                        help='progressive train ')
    parser.add_argument('--Use_tSNE', dest='Use_tSNE', action='store_false',
                        help='use tSNE show the penultimate Feature')
    parser.add_argument('--α',type=float, default=0.06, choices=[0.05],
                        help='BN penalty items between teacher(true) and teacher(fake)(default:0.07)')
    parser.add_argument('--β',type=float, default=0.8, choices=[0.5],
                        help='BN penalty items between teacher(true) and teacher(fake)(default:0.07)')
    parser.add_argument('--a_bit',type=int, default=1, choices=[1, 32],
                        help='quantile of activation')
    parser.add_argument('--w_bit',type=int, default=1, choices=[1, 32],
                        help='quantile of weight')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print(args)

    _, test_loader = get_dataloader(args)

    if args.dataset == 'cifar10':
        num_classes = 10 
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'caltech101':
        num_classes = 101
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200 
    teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes) # cifar10和cifar100所用教师网络
    # teacher = torchvision.models.resnet34(num_classes=101) # Caltech101所用教师网络
    student = network.resnet_8x.resnet18_1w1a(num_classes=num_classes, a_bit=args.a_bit, w_bit=args.w_bit) # 所用学生网络
    generator = network.gan.GeneratorA(nz=args.nz, nc=3, img_size=32) # CIFAR用生成器A,图片大小为32
    # generator = network.gan.GeneratorB(nz=args.nz, nc=3, img_size=128) # Caltech101用生成器B，图像大小为128

    teacher.load_state_dict(torch.load(args.ckpt, weights_only=True))
    print("Teacher restored from %s" % (args.ckpt))

    student.load_state_dict(torch.load('Ablation_Study/student/resnet18_base+BN+DA_2step_cifar10.pt', weights_only=True))

    generator.load_state_dict(torch.load('Ablation_Study/student/generator_resnet18_base+BN+DA_1step_cifar10.pt', weights_only=True))

    teacher.eval()

    teacher = teacher.to(device)
    student = student.to(device)
    generator = generator.to(device)

    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    if args.scheduler:
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, [100, 200], 0.1)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, [100, 200], 0.1)

    best_top1_acc = 0
    
    if args.test_only:
        top1_acc, top5_acc = test(args, student=student, generator=generator, device=device, test_loader=test_loader)
        return
    top1_acc_list = []
    top5_acc_list = []
    # Create hooks for teacher feature statistics
    loss_r_feature_layers = [] # 获取教师的BN层中数据的容器
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module)) # 获取了所有BN层中真图片与假图片的特征差距

    from network.resnet_8x import resnet20_1w1a, resnet18_1w1a, vgg_small_1w1a, resnet34_1w1a, resnet18_1w1a_ti
    model = torch.nn.DataParallel(eval(args.model)())
    model.cuda()

    def cpt_ab(epoch):
        "compute t&k in back-propagation"
        T_min, T_max = torch.tensor(args.Tmin).float(), torch.tensor(args.Tmax).float()
        Tmin, Tmax = torch.log10(T_min), torch.log10(T_max)
        a = torch.tensor([torch.pow(torch.tensor(10.), Tmin + (Tmax - Tmin) / args.epochs * epoch)]).float()
        b = max(1/t,torch.tensor(1.)).float()
        return a, b

    for epoch in range(1, args.epochs + 1):
        # Train
        if args.progressive:
            t = Log_UP(epoch, args.epochs)
            if (t < 1):
                k = 1 / t
            else:
                k = torch.tensor([1]).float().cuda()

            layer_cnt = 0
            param = []
            for m in model.modules():
                if isinstance(m, DA):
                    m.t = t
                    m.k = k
                    layer_cnt += 1

            a, b = cpt_ab(epoch)
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    module.b = b.cuda()
                    module.a = a.cuda()
            for module in model.modules():
                module.epoch = epoch

            line = f"layer : {layer_cnt}, k = {k.cpu().detach().numpy()[0]:.5f}, t = {t.cpu().detach().numpy()[0]:.5f}"
            # log.write("=> " + line + "\n")
            # log.flush()
            print(line)

        if args.scheduler:
            scheduler_S.step()
            scheduler_G.step()

        train(args, teacher=teacher, student=student, generator=generator, device=device,
              optimizer=[optimizer_S, optimizer_G], epoch=epoch, bn_hook=loss_r_feature_layers)# , t_hooks=t_hooks, tbn_stats=tbn_stats, s_hooks=s_hooks, sbn_stats=sbn_stats, SumWriter=SumWriter)
        # Test
        top1_acc, top5_acc = test(args, student, generator, device, test_loader, epoch)
        top1_acc_list.append(top1_acc)
        top5_acc_list.append(top5_acc)

        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            best_top5_acc = top5_acc

            torch.save(student.state_dict(),"Ablation_Study/student/test.pt")
            torch.save(generator.state_dict(), "Ablation_Study/student/generator_test.pt")
        # vp.add_scalar('Acc', epoch, acc)

    print("Best top1_Acc=%.6f" % best_top1_acc, "Best top5_Acc=%.6f" % best_top5_acc)


if __name__ == '__main__':
    main()