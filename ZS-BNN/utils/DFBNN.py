# 2022.09.29-Implementation for building AdaBin model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt
import numpy as np


# 特征图 直方图显示
def Hist_Show(x, name):
	"""
	输入:   网络某一层的输出特征(B, C , W, H)
	输出:   (0, 0 , W, H)对应的直方图
	"""
	x = x.detach()
	histNum = torch.reshape(input=x[0][0], shape=(1, -1)).cpu()
	# plt.hist(histNum, bins=range(-50, 50, 1))  # BN前
	plt.hist(histNum, bins=np.arange(-2, 2, 0.04))  # BN后
	plt.savefig(name)
	plt.close('all')

# 权重的前向二值函数与反向近似函数
class BinaryQuantize(Function):
	'''
		binary quantize function, from IR-Net
		(https://github.com/htqin/IR-Net/blob/master/CIFAR-10/ResNet20/1w1a/modules/binaryfunction.py)
	'''

	# 前向传播时，使用符号函数进行二值化
	@staticmethod
	def forward(ctx, input, k, t):
		ctx.save_for_backward(input, k, t)
		out = torch.sign(input)
		return out

	# 反向传播时，使用近似函数进行反向求导，grad_input = 近似函数的导数 * grad_output
	@staticmethod
	def backward(ctx, grad_output):
		input, k, t = ctx.saved_tensors
		k, t = k.cuda(), t.cuda()
		grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
		return grad_input, None, None

# 激活的前向二值函数与反向近似函数
class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input, b, a):
        ctx.save_for_backward(input, b, a)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, b, a = ctx.saved_tensors
        b = torch.tensor(1.)
        a = max(a, torch.tensor(1.))

        # input, k, t = ctx.saved_tensors
        # grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output

        grad_input = b * (2 * torch.sqrt(a**2 / 2) - torch.abs(a**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None

class Maxout(nn.Module):
	'''
		Nonlinear function
	'''

	def __init__(self, channel, neg_init=0.25, pos_init=1.0):
		super(Maxout, self).__init__()
		self.neg_scale = nn.Parameter(neg_init * torch.ones(channel))
		self.pos_scale = nn.Parameter(pos_init * torch.ones(channel))
		self.relu = nn.ReLU()

	def forward(self, x):
		# Maxout
		x = self.pos_scale.view(1, -1, 1, 1) * self.relu(x) - self.neg_scale.view(1, -1, 1, 1) * self.relu(-x)
		return x

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
	'''
		learnable distance and center for activation
	'''

	def __init__(self):
		super(BinaryActivation, self).__init__()
		# self.alpha_a = nn.Parameter(torch.tensor(1.0))
		# self.beta_a = nn.Parameter(torch.tensor(0.0))


		# 激活的均值，模长记录【记录个数 = 自定数 * Batch_Size】 --- 需在此单独修改:batch_size
		self.register_buffer('Mean_A', torch.tensor([0.0]))
		self.Mean_A_buffer = torch.zeros(1000, 256).cuda()
		self.Mean_A_index = 0
		self.register_buffer('alpha_A', torch.tensor([0.0]))
		self.alpha_A_buffer = torch.zeros(1000, 256).cuda()
		self.alpha_A_index = 0

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
		# x = (x - self.beta_a) / self.alpha_a
		# x = self.gradient_approx(x)
		# return self.alpha_a * (x + self.beta_a)

		###########################################################
		# 作用：减去均值，使信息熵最大化
		# 训练时：记录经过特征的均值，使用buffer区数据得到 self.Mean_A
		# 推理时：直接读取 self.Mean_A，进行推理
		if x.requires_grad:
			Mean_A = x.float().view(x.size(0), -1).mean(-1) #计算256张图片的均值
			self.Mean_A_buffer[self.Mean_A_index] = Mean_A.detach() # 将计算好的均值赋给第1行，一共1000行
			self.Mean_A_index = (self.Mean_A_index + 1) % 1000 # 索引加1
			self.Mean_A = torch.mean(self.Mean_A_buffer[self.Mean_A_buffer != 0]) # 对1000*256中的第一行求均值
			Mean_A = torch.mean(Mean_A.view(-1, 1, 1, 1))
			# Mean_A = Mean_A.view(-1, 1, 1, 1)
		else:
			Mean_A = self.Mean_A
		x = x - Mean_A

		# 作用：除以所有元素的均方根，使向量模长和二值参数一致
		# 训练时：记录经过特征的模长，使用buffer区数据得到 self.alpha_A
		# 推理时：直接读取 self.alpha_A，进行推理
		if x.requires_grad:
			alpha_A = torch.sqrt((x ** 2).sum((1, 2, 3)) / (x.size(1) * x.size(2) * x.size(3))) # α表示了全精度激活与二值化激活之间的模长差距
			self.alpha_A_buffer[self.alpha_A_index] = alpha_A.detach()
			self.alpha_A_index = (self.alpha_A_index + 1) % 1000
			self.alpha_A = torch.mean(self.alpha_A_buffer[self.alpha_A_buffer != 0])
			alpha_A = torch.mean(alpha_A.view(-1, 1, 1, 1))
			# alpha_A = alpha_A.view(-1, 1, 1, 1)
		else:
			alpha_A = self.alpha_A
		x = x / alpha_A # 全精度激活除以α，将全精度激活与二值激活的模长对齐

		x = self.gradient_approx(x)

		out = x * alpha_A + Mean_A # α+β，α-β，将激活二值为{a，b}

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


class AdaBin_Conv2d(nn.Conv2d):
	'''
		AdaBin Convolution
	'''

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
				 a_bit=1, w_bit=1):
		super(AdaBin_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
											bias)
		self.a_bit = a_bit
		self.w_bit = w_bit
		self.k = torch.tensor([10]).float().cpu()
		self.t = torch.tensor([0.1]).float().cpu()
		self.binary_a = BinaryActivation()
		self.filter_size = self.kernel_size[0] * self.kernel_size[1] * self.in_channels

	def forward(self, inputs):
		if self.a_bit == 1:
			inputs = self.binary_a(inputs) # 激活二值化

		if self.w_bit == 1:
			w = self.weight
			beta_w = w.mean((1, 2, 3)).view(-1, 1, 1, 1) # β通过均值求得
			alpha_w = torch.sqrt(((w - beta_w) ** 2).sum((1, 2, 3)) / self.filter_size).view(-1, 1, 1, 1) # α通过w-β的二范数，再除以c*k*k求得
			w = (w - beta_w) / alpha_w
			wb = BinaryQuantize().apply(w, self.k, self.t) # 二值{-1，+1}
			weight = wb * alpha_w + beta_w #二值{a, b}
		else:
			weight = self.weight

		output = F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

		return output

class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        bw = BinaryQuantize().apply(bw, self.k, self.t)
        ba = BinaryQuantize().apply(a, self.k, self.t)
        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

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
		loss = self.kl_criterion(log_probs, Soft_targets.detach()) * (temperature**2)
		return loss

class SCRM(nn.Module):
    """
    spatial & channel wise relation loss
    """
    def __init__(self, gamma=0.1):
        super(SCRM, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = gamma

    def spatial_wise(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = x.view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

    def channel_wise(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

    def cal_loss(self, f_s, f_t):
        f_s = F.normalize(f_s, dim=1)
        f_t = F.normalize(f_t, dim=1)
        sa_loss = F.l1_loss(self.spatial_wise(f_s), self.spatial_wise(f_t).detach())
        ca_loss = F.l1_loss(self.channel_wise(f_s), self.channel_wise(f_t).detach())
        return ca_loss + sa_loss

    def forward(self, g_s, g_t):
        return sum(self.cal_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t))


# 特征图匹配 空间通道损失函数
class Spatial_Channel_loss(nn.Module):
	"""
	输入:   需要进行匹配的特征图，匹配目标特征图 (二者需具有相同大小 (B, C , W, H) )
	操作：  compute distillation loss by referring to both the spatial and channel differences : Mean
	SVec： 空间向量             (B, 1, C)
	CVec： 维度矩阵(B, W, H) -> (B, 1, W*H)
	输出：  loss
	"""
	def __init__(self):
		super(Spatial_Channel_loss, self).__init__()
		self.SmoothL1 = nn.SmoothL1Loss(reduction='sum').cuda()

	def ghwloss(self, outputs, targets):
		assert outputs.size() == targets.size()

		# SVec_out = torch.sum(torch.sum(outputs, dim=3), dim=2)
		# SVec_tar = torch.sum(torch.sum(targets, dim=3), dim=2)
		# SVec_loss = torch.norm(SVec_out / torch.norm(SVec_out) - SVec_tar / torch.norm(SVec_tar))
		#
		# CVec_out = torch.sum(torch.sum(outputs, dim=1), dim=0)
		# CVec_tar = torch.sum(torch.sum(targets, dim=1), dim=0)
		# CVec_loss = torch.norm(CVec_out / torch.norm(CVec_out) - CVec_tar / torch.norm(CVec_tar))

		batch_size = targets.size()[0]
		channel_size = targets.size()[1]
		feature_size = targets.size()[2] ** 2

		SVec_out = torch.unsqueeze(torch.mean(torch.mean(outputs, dim=3), dim=2), dim=1)
		SVec_tar = torch.unsqueeze(torch.mean(torch.mean(targets, dim=3), dim=2), dim=1)
		SVec_out = SVec_out / torch.unsqueeze(torch.norm(SVec_out, dim=-1), dim=1).detach()
		SVec_tar = SVec_tar / torch.unsqueeze(torch.norm(SVec_tar, dim=-1), dim=1)

		CVec_out = torch.unsqueeze(torch.mean(outputs, dim=1).reshape(batch_size, -1), dim=1)
		CVec_tar = torch.unsqueeze(torch.mean(targets, dim=1).reshape(batch_size, -1), dim=1)
		CVec_out = CVec_out / torch.unsqueeze(torch.norm(CVec_out, dim=-1), dim=1).detach()
		CVec_tar = CVec_tar / torch.unsqueeze(torch.norm(CVec_tar, dim=-1), dim=1)

		# # DGRL论文 空间通道注意力loss
		SVec_loss = torch.norm(SVec_out - SVec_tar) / channel_size
		CVec_loss = torch.norm(CVec_out - CVec_tar) / feature_size
		loss = (SVec_loss + CVec_loss) / batch_size
		# SVec_loss = self.SmoothL1(SVec_out, SVec_tar) / channel_size
		# CVec_loss = self.SmoothL1(CVec_out, CVec_tar) / feature_size
		# loss = (SVec_loss + CVec_loss) / batch_size

		# # 余弦相似度  空间通道注意力loss
		# SVec_CosSim_loss = (SVec_out * SVec_tar).sum(dim=-1)
		# SVec_CosSim_loss = ((torch.ones_like(SVec_CosSim_loss) - SVec_CosSim_loss) * 0.5).sum() / channel_size
		# CVec_CosSim_loss = (CVec_out * CVec_tar).sum(dim=-1)
		# CVec_CosSim_loss = ((torch.ones_like(CVec_CosSim_loss) - CVec_CosSim_loss) * 0.5).sum() / feature_size
		# loss = (SVec_CosSim_loss + CVec_CosSim_loss) / batch_size

		return loss

	def forward(self, g_s, g_t):
		return sum(self.ghwloss(f_s, f_t) for f_s, f_t in zip(g_s, g_t))


class SimKD(nn.Module):
	"""CVPR-2022: Knowledge Distillation with the Reused Teacher Classifier"""

	def __init__(self, *, s_n, t_n, factor=2):
		super(SimKD, self).__init__()

		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

		def conv1x1(in_channels, out_channels, stride=1):
			return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

		def conv3x3(in_channels, out_channels, stride=1, groups=1):
			return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False,
							 groups=groups)

		# A bottleneck design to reduce extra parameters
		setattr(self, 'transfer', nn.Sequential(
			conv1x1(s_n, t_n // factor),
			nn.BatchNorm2d(t_n // factor),
			nn.ReLU(inplace=True),
			conv3x3(t_n // factor, t_n // factor),
			# depthwise convolution
			# conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
			nn.BatchNorm2d(t_n // factor),
			nn.ReLU(inplace=True),
			conv1x1(t_n // factor, t_n),
			nn.BatchNorm2d(t_n),
			nn.ReLU(inplace=True),
		))

	def forward(self, feat_s, feat_t, cls_t):

		# Spatial Dimension Alignment
		s_H, t_H = feat_s.shape[2], feat_t.shape[2]
		if s_H > t_H:
			source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
			target = feat_t
		else:
			source = feat_s
			target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))

		trans_feat_t = target

		# Channel Alignment
		trans_feat_s = getattr(self, 'transfer')(source)

		# Prediction via Teacher Classifier
		temp_feat = self.avg_pool(trans_feat_s)
		temp_feat = temp_feat.view(temp_feat.size(0), -1)
		pred_feat_s = cls_t(temp_feat)

		return trans_feat_s, trans_feat_t, pred_feat_s