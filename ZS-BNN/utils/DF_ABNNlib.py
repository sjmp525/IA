import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import ortho_group
import copy
from utils.zsbnn_1w1a import BinaryQuantize_STE, BinaryQuantize_ours, BinaryQuantize_RBNN, BinaryQuantize_RBNN_a, BinaryQuantize_IR, BinaryQuantize_ReCU, BinaryQuantize_ReCU_a, BinaryQuantize_BiPer, BinaryQuantize_BiPer_a
import seaborn as sns
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

# 特征图 直方图显示
def Hist_Show(x, name, bins, color, edgecolor):
    """
    输入:   网络某一层的输出特征(B, C , W, H)
    输出:   (0, 0 , W, H)对应的直方图
    """
    x = x.detach()  # 解除计算图
    histNum = torch.reshape(x, shape=(-1,)).cpu()  # 展平为一维数组
    plt.figure(figsize=(8.5, 6))

    # 生成一个整体的直方图
    plt.hist(histNum, bins=bins, color=color, alpha=0.7, edgecolor=edgecolor)
    plt.xticks(np.arange(-4, 5, 2), fontsize=30)
    plt.yticks(fontsize=30)

    plt.savefig(name)
    plt.close('all')

def Hist_Show_KDE(x, name, color):
    """
    输入:   网络某一层的输出特征(B, C , W, H)
    输出:   对应的概率密度曲线
    """
    x = x.detach()  # 从计算图中分离张量
    data = torch.reshape(input=x, shape=(1, -1)).cpu().numpy()  # 将张量展平成1D

    plt.figure(figsize=(8.5, 6))

    # 使用 fill=True 代替 shade
    sns.kdeplot(data[0], fill=True, color=color, linewidth=3)

    plt.xticks(np.arange(-2, 3, 1), fontsize=22)
    plt.yticks(fontsize=22)
    # plt.title('Probability Density Function', fontsize=18)
    plt.xlabel('Value', fontsize=18)
    plt.ylabel('Density', fontsize=18)

    plt.savefig(name)
    plt.close('all')

class RPReLU(nn.Module):
    """
		Nonlinear function
		Copy from Paper: ReActNet
	"""

    def __init__(self, channel):
        super(RPReLU, self).__init__()
        self.bias1 = nn.Parameter(torch.zeros(1, channel, 1, 1), requires_grad=True)
        self.prelu = nn.PReLU(channel)
        self.bias2 = nn.Parameter(torch.zeros(1, channel, 1, 1), requires_grad=True)

    def forward(self, x):
        x = x + self.bias1  # .expand_as(x)
        x = self.prelu(x)
        x = x + self.bias2
        return x

class BinaryActivation(nn.Module):

    def __init__(self):
        super(BinaryActivation, self).__init__()
        # self.alpha_a = nn.Parameter(torch.tensor(1.0))
        # self.beta_a = nn.Parameter(torch.tensor(0.0))

        # 激活的均值，模长记录【记录个数 = 自定数 * Batch_Size】 --- 需在此单独修改:batch_size
        self.register_buffer('Mean_A', torch.tensor([0.0]))
        self.Mean_A_buffer = torch.zeros(250, 256).cuda()
        self.Mean_A_index = 0
        self.register_buffer('alpha_A', torch.tensor([1.0]))
        self.alpha_A_buffer = torch.ones(250, 256).cuda()
        self.alpha_A_index = 0

        self.Mean_A = torch.tensor(0.0)
        self.alpha_A = torch.tensor(1.0)

        self.b = torch.tensor([10.]).float()
        self.a = torch.tensor([0.1]).float()

    def gradient_approx(self, x):
        '''
			gradient approximation
			(https://github.com/liuzechun/Bi-Real-net/blob/master/pytorch_implementation/BiReal18_34/birealnet.py)
		'''
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

    def forward(self, x):
        ###########################################################
        # 作用：减去均值，使信息熵最大化
        # 训练时：记录经过特征的均值，使用buffer区数据得到 self.Mean_A
        # 推理时：直接读取 self.Mean_A，进行推理
        if x.requires_grad:
            # Mean_A = x.float().view(x.size(0), -1).mean(-1)  # 计算256张图片的均值
            mx = x.view(x.shape[0], -1)
            Mean_A = torch.median(mx, dim=1).values # 中位数
            self.Mean_A_buffer[self.Mean_A_index] = Mean_A.detach()  # 将计算好的均值赋给第1行，一共1000行
            self.Mean_A_index = (self.Mean_A_index + 1) % 250  # 索引加1
            self.Mean_A = torch.mean(self.Mean_A_buffer[self.Mean_A_buffer != 0])  # 对1000*256中的第一行求均值
            Mean_A = torch.mean(Mean_A.view(-1, 1, 1, 1))
        # Mean_A = Mean_A.view(-1, 1, 1, 1)
        else:
            Mean_A = self.Mean_A
        x = x - Mean_A

        # 作用：除以所有元素的均方根，使向量模长和二值参数一致
        # 训练时：记录经过特征的模长，使用buffer区数据得到 self.alpha_A
        # 推理时：直接读取 self.alpha_A，进行推理
        if x.requires_grad:
            alpha_A = torch.sqrt(
                (x ** 2).sum((1, 2, 3)) / (x.size(1) * x.size(2) * x.size(3)))  # α表示了全精度激活与二值化激活之间的模长差距
            self.alpha_A_buffer[self.alpha_A_index] = alpha_A.detach()
            self.alpha_A_index = (self.alpha_A_index + 1) % 250
            self.alpha_A = torch.mean(self.alpha_A_buffer[self.alpha_A_buffer != 0])
            alpha_A = torch.mean(alpha_A.view(-1, 1, 1, 1))
        # alpha_A = alpha_A.view(-1, 1, 1, 1)
        else:
            alpha_A = self.alpha_A
        x = x / alpha_A  # 全精度激活除以α，将全精度激活与二值激活的模长对齐

        x = self.gradient_approx(x)

        out = x * alpha_A + Mean_A  # α+β，α-β，将激活二值为{a，b}

        return out

class LambdaLayer(nn.Module):
    '''
		for DownSample
	'''

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class DA(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 a_bit=1, w_bit=1):
        super(DA, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                 bias)
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.binary_a = BinaryActivation()
        self.filter_size = self.kernel_size[0] * self.kernel_size[1] * self.in_channels

    def forward(self, inputs):
        if self.a_bit == 1:
            # 分布对齐
            inputs = self.binary_a(inputs)  # 激活二值化
        else:
            inputs = inputs

        if self.w_bit == 1:
            w = self.weight
            # 分布对齐
            mw = w.view(w.shape[0], -1)
            beta_w = torch.median(mw, dim=1).values.view(-1, 1, 1, 1) # 中位数
            # beta_w = w.mean((1, 2, 3)).view(-1, 1, 1, 1)  # 均值
            alpha_w = torch.sqrt(((w - beta_w) ** 2).sum((1, 2, 3)) / self.filter_size).view(-1, 1, 1, 1)  # α通过w-β的二范数，再除以c*k*k求得
            w = (w - beta_w) / alpha_w
            wb = BinaryQuantize_ours().apply(w, self.k, self.t)  # 二值{-1，+1}
            weight = wb * alpha_w + beta_w  # 二值{a, b}

        else:
            weight = self.weight

        output = F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

class BinarizeConv2d_BNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 a_bit=32, w_bit=1):
        super(BinarizeConv2d_BNN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                 bias)
        self.a_bit = a_bit
        self.w_bit = w_bit

    def forward(self, inputs):
        if self.a_bit == 1:
            inputs =  BinaryQuantize_STE().apply(inputs, torch.tensor([1.0]).cuda(), torch.tensor([1.0]).cuda())  # 激活二值化
        else:
            inputs = inputs

        if self.w_bit == 1:
            w = self.weight
            weight = BinaryQuantize_STE().apply(w, torch.tensor([1.0]).cuda(), torch.tensor([1.0]).cuda())
        else:
            weight = self.weight

        output = F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

class IRConv2d(nn.Conv2d):
    'IR-Net'

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()
        bw = BinaryQuantize_IR().apply(bw, self.k, self.t)
        ba = BinaryQuantize_IR().apply(a, self.k, self.t)
        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        return output

class BinarizeConv2d_RBNN(nn.Conv2d):
    'RBNN'

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d_RBNN, self).__init__(*kargs, **kwargs)
        self.k = torch.tensor([10.]).float()
        self.t = torch.tensor([0.1]).float()
        self.epoch = -1

        w = self.weight
        self.a, self.b = get_ab(np.prod(w.shape[1:]))
        R1 = torch.tensor(ortho_group.rvs(dim=self.a)).float().cuda()
        R2 = torch.tensor(ortho_group.rvs(dim=self.b)).float().cuda()
        self.register_buffer('R1', R1)
        self.register_buffer('R2', R2)
        self.Rweight = torch.ones_like(w)

        sw = w.abs().view(w.size(0), -1).mean(-1).float().view(w.size(0), 1, 1).detach()
        self.alpha = nn.Parameter(sw.cuda(), requires_grad=True)
        self.rotate = nn.Parameter(torch.ones(w.size(0), 1, 1, 1).cuda() * np.pi / 2, requires_grad=True)
        self.Rotate = torch.zeros(1)

    def forward(self, input):
        a0 = input
        w = self.weight
        w1 = w - w.mean([1, 2, 3], keepdim=True)
        w2 = w1 / w1.std([1, 2, 3], keepdim=True)
        a1 = a0 - a0.mean([1, 2, 3], keepdim=True)
        a2 = a1 / a1.std([1, 2, 3], keepdim=True)

        a, b = self.a, self.b
        X = w2.view(w.shape[0], a, b)
        if self.epoch > -1 and self.epoch % 1 == 0:
            for _ in range(3):
                # * update B
                V = self.R1.t() @ X.detach() @ self.R2
                B = torch.sign(V)
                # * update R1
                D1 = sum([Bi @ (self.R2.t()) @ (Xi.t()) for (Bi, Xi) in zip(B, X.detach())])
                U1, S1, V1 = torch.svd(D1)
                self.R1 = (V1 @ (U1.t()))
                # * update R2
                D2 = sum([(Xi.t()) @ self.R1 @ Bi for (Xi, Bi) in zip(X.detach(), B)])
                U2, S2, V2 = torch.svd(D2)
                self.R2 = (U2 @ (V2.t()))
        self.Rweight = ((self.R1.t()) @ X @ (self.R2)).view_as(w)
        delta = self.Rweight.detach() - w2
        w3 = w2 + torch.abs(torch.sin(self.rotate)) * delta

        # * binarize
        bw = BinaryQuantize_RBNN().apply(w3, self.k.to(w.device), self.t.to(w.device))
        ba = BinaryQuantize_RBNN_a().apply(a2, self.k.to(w.device), self.t.to(w.device))
        # * 1bit conv
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        # * scaling factor
        output = output * self.alpha
        return output

class BinarizeConv2d_ReCU(nn.Conv2d):
    'ReCU'

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d_ReCU, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))

    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - w.mean([1, 2, 3], keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var([1, 2, 3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
        EW = torch.mean(torch.abs(w1))
        Q_tau = (- EW * torch.log(2 - 2 * self.tau)).detach().cpu().item()
        w2 = torch.clamp(w1, -Q_tau, Q_tau)

        if self.training:
            a0 = a / torch.sqrt(a.var([1, 2, 3], keepdim=True) + 1e-5)
        else:
            a0 = a

        # * binarize
        bw = BinaryQuantize_ReCU().apply(w2)
        ba = BinaryQuantize_ReCU_a().apply(a0)
        # * 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        # * scaling factor
        output = output * self.alpha
        return output

class BinarizeConv2d_BiPer(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d_BiPer, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(self.weight.size(0), 1, 1), requires_grad=False)
        self.register_buffer('tau', torch.tensor(1.))

        self.freq = 30
        # self.stage2 = args.stage2
        # print(f"using stage2: {self.stage2}")
        # print(self.freq)

    def forward(self, input):
        a = input
        w = self.weight

        if self.training:
            a0 = a / torch.sqrt(a.var([1, 2, 3], keepdim=True) + 1e-5)
        else:
            a0 = a

		# * binarize
        w = torch.sin(self.freq * w)
        bw = BinaryQuantize_BiPer().apply(w)
        ba = BinaryQuantize_BiPer_a().apply(a0)

		# * 1+bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
		# * scaling factor
        output = output * self.alpha
        return output

def get_ab(N):
    sqrt = int(np.sqrt(N))
    for i in range(sqrt, 0, -1):
        if N % i == 0:
            return i, N // i

def add_current_data_feature(teacher_model, last_feature_dict, hooks, old_data, old_label):
    with torch.no_grad():
        teacher_model.eval()
        bs = 64

        for i in range(len(old_data) // bs):
            fake_images, labels = old_data[i * bs:(i + 1) * bs], old_label[i * bs:(i + 1) * bs]
            inp = torch.from_numpy(fake_images).cuda()
            inp_labels = torch.from_numpy(labels).cuda()
            for hook in hooks:
                hook.clear()
            _, output = teacher_model(inp)
            last_features = hook.outputs
            gt = inp_labels.data.cpu().numpy()

            # d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
            # print(d_acc)

            for j in range(len(gt)):
                l = gt[j]
                last_feature = last_features[j]
                if l not in last_feature_dict:
                    last_feature_dict[l] = []
                last_feature_dict[l].append(copy.deepcopy(torch.squeeze(last_feature).data.cpu().numpy()))
    return last_feature_dict

def Log_UP(epoch, total_epochs):
    K_min, K_max = 1e-3, 1e1
    # K_min, K_max = 1e-1, 1e1
    Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / total_epochs * epoch)]).float().cuda()

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn) # 容器中存储假图片与真图片的特征差距

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2) # 假图片与真图片的特征差距：L2范数计算：假图片方差（在更新）-真图片方差（不变） + 假图片均值（在更新）-真图片均值（不变）(反了)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

class KL_SoftLabelloss(nn.Module):
    """
	输入:   网络输出(不需要提前softmax)，one-hot标签(和必须为1)
	操作：  torch.softmax(true_outputs, dim=1)
	输出：  loss
	"""

    def __init__(self):
        super(KL_SoftLabelloss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.temperature = 4

    def forward(self, outputs, targets, temperature):
        log_probs = self.logsoftmax(outputs / temperature)
        Soft_targets = self.softmax(targets / temperature)
        loss = self.kl_criterion(log_probs, Soft_targets.detach()) * (temperature ** 2)
        return loss
    