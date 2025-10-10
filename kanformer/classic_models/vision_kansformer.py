"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
# 项目讲解链接：https://www.bilibili.com/video/BV1px4y1W7zs/?spm_id_from=333.999.0.0&vd_source=6528929ff3772e61e9f5baf9b8ab1e64
from functools import partial # 从 functools 模块中导入 partial 函数
from collections import OrderedDict # 从 collections 模块中导入 OrderedDict 类
# import sys 
# sys.path.append(r"D:\code\Kansformer")
from src.efficient_kan import KAN
import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    在每个样本上应用 Stochastic Depth（随机深度）来丢弃路径（当应用于残差块的主路径时）。
    这与EfficientNet等网络创建的 DropConnect 实现相同，但是，原始名称有误导性，因为'Drop Connect' 是另一篇论文中不同形式的 dropout...
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择了
    将层和参数名称更改为 'drop path'，而不是将 DropConnect 作为层名称并使用 'survival rate' 作为参数。
    """
    if drop_prob == 0. or not training: # 如果 drop_prob 为 0 或者模型不处于训练模式，直接返回输入张量 x，不进行任何操作
        return x
    keep_prob = 1 - drop_prob # 计算保持路径的概率，即 1 减去丢弃路径的概率
    # 创建一个与输入张量 x 的形状兼容的 shape
    # (x.shape[0],) 表示保持批次维度，(1,) * (x.ndim - 1) 表示在其他维度上保持单一值
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    # 创建一个与输入张量形状兼容的随机张量 random_tensor
    # 这个张量的值在 keep_prob 和 1 之间（包含 keep_prob）
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # 使用 floor_() 方法将随机张量中的值二值化
    # 所有大于等于 1 的值将变为 1，所有小于 1 的值将变为 0
    random_tensor.floor_()  
    # 将输入张量 x 按照 keep_prob 进行缩放（除以 keep_prob）
    # 然后与二值化的 random_tensor 相乘，以实现按比例丢弃路径
    output = x.div(keep_prob) * random_tensor
    return output # 返回处理后的输出张量


class DropPath(nn.Module):
    """
    用于每个样本的Drop paths（随机深度）（当应用于残差块的主路径时）。
    """
    def __init__(self, drop_prob=None):
        """
        初始化 DropPath 类。

        Args:
            drop_prob: 丢弃路径的概率，默认为 None。
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入张量。

        Returns:
            经过 DropPath 操作后的张量。
        """
        # 调用 drop_path 函数，传入输入张量 x、丢弃路径概率 self.drop_prob 和训练模式 self.training
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    将2D图像转换为Patch嵌入
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        """
        初始化PatchEmbed类。

        Args:
            img_size: 输入图像的尺寸，默认为224。
            patch_size: Patch的尺寸，默认为16。
            in_c: 输入图像的通道数，默认为3。
            embed_dim: 嵌入的维度，默认为768。
            norm_layer: 规范化层，默认为None。
        """
        super().__init__()
        # 设置输入图像和Patch的尺寸
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 计算图像的网格尺寸和Patch数量
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # 224/16=14
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 使用卷积层进行投影
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
         # 应用规范化层
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入张量，形状为[B, C, H, W]，表示批次大小、通道数、高度和宽度。

        Returns:
            嵌入的张量，形状为[B, num_patches, embed_dim]，表示批次大小、Patch数量和嵌入维度。
        """
        B, C, H, W = x.shape  # 获取输入张量的形状信息
        # 检查输入图像尺寸是否与预期的尺寸相匹配
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 使用卷积层进行投影，并将结果展平并转置以匹配期望的输出形状
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x) # 应用规范化层
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8, # 注意力头的数量，默认为8
                 qkv_bias=False,  # 是否为注意力查询、键、值添加偏置，默认为False
                 qk_scale=None, # 查询和键的缩放因子，默认为 None
                 attn_drop_ratio=0., # 注意力权重的丢弃率，默认为0
                 proj_drop_ratio=0.):  # 输出投影的丢弃率，默认为0
        super(Attention, self).__init__()
        # 初始化注意力层
        self.num_heads = num_heads # 设置注意力头的数量
        head_dim = dim // num_heads # 计算每个头的维度
        self.scale = qk_scale or head_dim ** -0.5 # 设置缩放因子，若未提供则默认为头维度的倒数的平方根
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 初始化注意力机制中的查询、键、值的线性变换
        self.attn_drop = nn.Dropout(attn_drop_ratio) # 初始化用于丢弃注意力权重的Dropout层
        self.proj = nn.Linear(dim, dim) # 初始化输出投影的线性变换
        self.proj_drop = nn.Dropout(proj_drop_ratio) # 初始化用于丢弃输出投影的Dropout层

    def forward(self, x):
        # 获取输入张量 x 的形状信息
        # x: [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
         # 对输入张量 x 进行线性变换得到查询、键、值
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 将查询、键、值分离出来
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # 计算注意力权重
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x) # 对输出进行线性变换
        x = self.proj_drop(x) # 对输出进行 Dropout 操作
        return x


class Mlp(nn.Module):
    """
    在Vision Transformer、MLP-Mixer和相关网络中使用的MLP
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        初始化MLP类。

        Args:
            in_features: 输入特征的维度。
            hidden_features: 隐藏层特征的维度，默认为输入特征的维度。
            out_features: 输出特征的维度，默认为输入特征的维度。
            act_layer: 激活函数，默认为GELU激活函数。
            drop: Dropout层的丢弃率，默认为0。
        """
        super().__init__()
        # 设置隐藏层和输出层特征维度，默认与输入特征维度相同
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 初始化第一个全连接层、激活函数和第二个全连接层
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)  # 初始化Dropout层

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    在Vision Transformer中使用的基本块
    """
    """
    初始化Block类。

    Args:
        dim: 输入维度。
        num_heads: 注意力头的数量。
        mlp_ratio: MLP隐藏层维度与输入维度的比率，默认为4。
        qkv_bias: 是否为注意力查询、键、值添加偏置，默认为False。
        qk_scale: 查询和键的缩放因子，默认为None。
        drop_ratio: 通用的丢弃率，默认为0。
        attn_drop_ratio: 注意力权重的丢弃率，默认为0。
        drop_path_ratio: DropPath操作的丢弃率，默认为0。
        act_layer: 激活函数，默认为GELU激活函数。
        norm_layer: 规范化层，默认为LayerNorm。
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        # 初始化第一个规范化层和注意力机制
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 初始化DropPath操作
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        # 初始化第二个规范化层和KAN
        self.norm2 = norm_layer(dim)
        self.kan = KAN([dim,64,dim])
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入张量。

        Returns:
            经过Block操作后的张量。
        """
        b,t,d = x.shape # 获取输入张量 x 的批次大小、序列长度和特征维度
        # 对输入张量进行规范化、注意力机制、DropPath操作
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # 对输入张量进行规范化、KAN操作、DropPath操作
        # 对输入张量进行规范化，然后将其形状重塑为 [b*t, d]，其中 b 为批次大小，t 为序列长度，d 为特征维度
        # 将重塑后的张量再次重塑为 [b, t, d] 的形状，恢复原始的批次大小、序列长度和特征维度
        # x = x + self.drop_path(self.kan(self.norm2(x)))
        x = x + self.drop_path(self.kan(self.norm2(x).reshape(-1,x.shape[-1])).reshape(b,t,d))

        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        初始化 Vision Transformer 模型。

        Args:
            img_size (int, tuple): 输入图像大小。
            patch_size (int, tuple): patch 大小。
            in_c (int): 输入通道数。
            num_classes (int): 分类头中的类别数。
            embed_dim (int): 嵌入维度。
            depth (int): Transformer 的深度。
            num_heads (int): 注意力头的数量。
            mlp_ratio (int): MLP 隐藏维度与嵌入维度的比率。
            qkv_bias (bool): 是否为注意力查询、键、值添加偏置。
            qk_scale (float): 查询和键的缩放因子。
            representation_size (Optional[int]): 如果设置，则启用并设置表示层（预对数层）的大小。
            distilled (bool): 模型是否包含蒸馏标记和头部，如 DeiT 模型。
            drop_ratio (float): dropout 比率。
            attn_drop_ratio (float): 注意力 dropout 比率。
            drop_path_ratio (float): 随机深度率。
            embed_layer (nn.Module): patch 嵌入层。
            norm_layer: (nn.Module): 规范化层。
            act_layer: 激活函数。
        """
        super(VisionTransformer, self).__init__()
        # 初始化基本参数和层
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # 初始化 patch 嵌入层
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # 初始化类别标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # 初始化随机深度率
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # 随机深度衰减规则
        # 初始化 Transformer 块
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # 初始化表示层
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # 初始化分类器头 将线性层更换为KAN
        self.head = KAN([self.num_features, 64,num_classes]) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        """
        前向传播，提取特征部分。
        Args:
            x (tensor): 输入张量，形状为 [B, C, H, W]，其中 B 表示批次大小，C 表示通道数，H 和 W 表示输入图像的高度和宽度。
        Returns:
            tensor: 提取的特征张量，形状为 [B, num_patches, embed_dim]。
        """
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # 将输入张量通过 patch 嵌入层转换为形状为 [B, num_patches, embed_dim] 的特征张量
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        # 将类别标记扩展为与特征张量相同的形状，形状为 [B, 1, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 将类别标记与特征张量拼接在一起，形状为 [B, num_patches + num_tokens, embed_dim]
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed) # 对拼接后的张量进行位置编码并进行 dropout
        x = self.blocks(x) # 经过 Transformer 块处理特征张量
        x = self.norm(x)  # 对处理后的特征张量进行规范化
        if self.dist_token is None: # 返回处理后的特征张量
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        """
        完整的前向传播。

        Args:
            x (tensor): 输入张量。

        Returns:
            tensor: 输出张量。
        """
        x = self.forward_features(x) # 提取特征
        if self.head_dist is not None: # 如果存在蒸馏头部，分别计算主分类器和蒸馏分类器的预测结果
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist  # 训练时返回主分类器和蒸馏分类器的预测结果
            else:
                return (x + x_dist) / 2 # 推理时返回主分类器和蒸馏分类器预测结果的平均值
        else:
            x = self.head(x) # 否则只使用主分类器预测结果
        return x


def _init_vit_weights(m):
    """
    ViT 权重初始化函数。

    Args:
        m (Module): 模型中的模块。
    """
    if isinstance(m, nn.Linear):
        # 如果是线性层，使用截断的正态分布初始化权重，标准差为 0.01
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # 如果是二维卷积层，使用 Kaiming 正态分布初始化权重
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        # 如果是 LayerNorm 层，初始化偏置为零，权重为一
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def kit_base_patch16_224(num_classes: int = 1000): # 构建 KiT-Base 模型
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224, # 输入图像的大小，为 224x224
                              patch_size=16, # 感受野大小，即每个patch的大小为16x16
                              embed_dim=768,  # 嵌入维度，即 Transformer 模型中每个token的维度
                              depth=12, # Transformer 模型的层数
                              num_heads=12,  # 注意力头数，即每个注意力层中多头注意力的头数
                              representation_size=None, # 表示层的大小，用于控制模型输出的维度，如果为 None，则不进行降维处理
                              num_classes=num_classes) # 分类的类别数
    return model


def kit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def kit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def kit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def kit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def kit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def kit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def kit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model