import torch
from torch.autograd import Function
import numpy as np

#############################################STE
class BinaryQuantize_STE(Function):
    """
        反向传播近似函数：...
        可变HardTanh近似函数，高度aa，可更新范围(-1/bb, 1/bb)
    """
    @staticmethod
    def forward(ctx, input, aa, bb):
        ctx.save_for_backward(input, aa, bb)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, aa, bb = ctx.saved_tensors
        grad_input = (aa * bb) * grad_output
        grad_input[input.gt(1./bb)] = 0
        grad_input[input.lt(-1./bb)] = 0
        return grad_input, None, None
#######################################STE

########################################Adabin
class BinaryQuantize_ours(Function):
    '''
        binary quantize function, from IR-Net
        (https://github.com/htqin/IR-Net/blob/master/CIFAR-10/ResNet20/1w1a/modules/binaryfunction.py)
    '''

    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        out[input == 0] = 1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k, t = k.cuda(), t.cuda()
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

#########################################RBNN的近似函数
class BinaryQuantize_RBNN(Function):
	@staticmethod
	def forward(ctx, input, k, t):
		ctx.save_for_backward(input, k, t)
		out = torch.sign(input)
		return out

	@staticmethod
	def backward(ctx, grad_output):
		input, k, t = ctx.saved_tensors
		grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
		grad_input = grad_input.clamp(min=0) * grad_output.clone()
		return grad_input, None, None


class BinaryQuantize_RBNN_a(Function):
	@staticmethod
	def forward(ctx, input, k, t):
		ctx.save_for_backward(input, k, t)
		out = torch.sign(input)
		return out

	@staticmethod
	def backward(ctx, grad_output):
		input, k, t = ctx.saved_tensors
		k = torch.tensor(1.).to(input.device)
		t = max(t, torch.tensor(1.).to(input.device))
		grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
		grad_input = grad_input.clamp(min=0) * grad_output.clone()
		return grad_input, None, None
#########################################RBNN的近似函数
#
def get_ab(N):
	sqrt = int(np.sqrt(N))
	for i in range(sqrt, 0, -1):
		if N % i == 0:
			return i, N // i


####################################IR-Net的近似函数
class BinaryQuantize_IR(Function):

    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        # k, t = k.cuda(), t.cuda()
        grad_input = k * t * (1-torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None
####################################IR-Net的近似函数

#######################ReCU的近似函数
class BinaryQuantize_ReCU(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class BinaryQuantize_ReCU_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input
######################ReCU的近似函数

######################################BiPer的近似函数
class BinaryQuantize_BiPer(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_BiPer_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input
######################################BiPer的近似函数