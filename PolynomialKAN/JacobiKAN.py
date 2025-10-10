import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    """
    KANLinear with Jacobi polynomial basis instead of Gaussian kernels.

    - 使用 Jacobi 多项式基（阶数 = grid_size）代替原高斯核。
    - 可设 alpha, beta（须 > -1），并支持输入缩放：'tanh' 或按批 min-max 线性缩放到 [-1,1]。
    - 为避免反传中的 AsStridedBackward/in-place 问题，关键位置添加了 .contiguous() / reshape，禁用任何原地激活。
    - 保留 update_grid 接口（无操作），保证与原训练循环兼容。
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,                 # Jacobi degree n
        scale_noise=0.1,
        scale_base=1.0,
        scale_kernel=1.0,            # 兼容参数（不参与 Jacobi 评估）
        enable_standalone_scale_kernel=True,  # 仍用于系数缩放
        base_activation=torch.nn.SiLU,        # 非原地版本
        grid_eps=0.02,               # 兼容参数
        grid_range=[-1, 1],          # 若使用 'linear' ，可自定义固定范围（默认按批 min-max）
        alpha=0.0,                   # Jacobi alpha > -1
        beta=0.0,                    # Jacobi beta  > -1
        input_scaling='tanh',        # 'tanh' or 'linear'
    ):
        super(KANLinear, self).__init__()
        assert alpha > -1 and beta > -1, "Jacobi parameters must satisfy alpha,beta > -1"

        self.in_features = in_features
        self.out_features = out_features
        self.degree = int(grid_size)      # n
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.input_scaling = input_scaling
        self.grid_range = grid_range

        # Base 分支（与原实现一致）
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # Jacobi 基系数: (out_features, in_features, degree+1)
        self.kernel_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, self.degree + 1)
        )

        self.enable_standalone_scale_kernel = enable_standalone_scale_kernel
        if enable_standalone_scale_kernel:
            self.kernel_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_kernel = scale_kernel
        self.base_activation = base_activation()  # 默认为非原地 SiLU
        self.grid_eps = grid_eps  # 兼容保留

        self.reset_parameters()

    def reset_parameters(self):
        # Base 线性层初始化
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # 用 Jacobi 基最小二乘投影到随机小噪声曲线以初始化 kernel_weight
        with torch.no_grad():
            # 采样点 S（足够多以稳定 lstsq）
            S = max(2 * (self.degree + 1), 16)
            z = torch.linspace(-1.0, 1.0, S).unsqueeze(1).repeat(1, self.in_features).contiguous()  # (S, in_features)

            # 目标“曲线”噪声: (S, in_features, out_features)
            noise = (torch.rand(S, self.in_features, self.out_features) - 0.5) * (self.scale_noise / max(1, self.degree))
            noise = noise.contiguous()

            # 构建 Jacobi 基: (S, in_features, K)
            A = self.jacobi_basis(z).contiguous()

            # 逐个输入特征做最小二乘，得到 (out, in, K)
            K = self.degree + 1
            coeff = torch.zeros(self.out_features, self.in_features, K)
            for i in range(self.in_features):
                Ai = A[:, i, :]               # (S, K)
                Bi = noise[:, i, :]           # (S, out)
                Ci = torch.linalg.lstsq(Ai, Bi).solution  # (K, out)
                coeff[:, i, :] = Ci.T  # (out, K)

            self.kernel_weight.copy_(coeff)

            if self.enable_standalone_scale_kernel:
                torch.nn.init.kaiming_uniform_(self.kernel_scaler, a=math.sqrt(5) * self.scale_kernel)

    @staticmethod
    def _tanh_scale(x: torch.Tensor) -> torch.Tensor:
        # 非原地，映射到 [-1,1]
        return torch.tanh(x)

    @staticmethod
    def _linear_scale_batch(x: torch.Tensor) -> torch.Tensor:
        # 按批、按通道 min-max -> [-1,1]，避免除零并确保连续内存
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        denom = (x_max - x_min).clamp_min(1e-6)
        out = ((x - x_min) / denom) * 2.0 - 1.0
        return out

    def _scale_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scaling == 'tanh':
            return self._tanh_scale(x)
        elif self.input_scaling == 'linear':
            # 默认按批 min-max；如需固定范围，可按 grid_range 做线性缩放
            return self._linear_scale_batch(x)
        else:
            raise ValueError("input_scaling must be 'tanh' or 'linear'")

    def jacobi_basis(self, z: torch.Tensor) -> torch.Tensor:
        """
        构建 Jacobi 基: [P_0^{(a,b)}(z), ..., P_n^{(a,b)}(z)]
        z: (B, in_features)
        return: (B, in_features, degree+1)
        """
        a, b = self.alpha, self.beta
        assert z.dim() == 2
        B, F = z.shape
        K = self.degree + 1

        P = z.new_zeros(B, F, K).contiguous()
        # P0
        P[..., 0] = 1.0

        if self.degree >= 1:
            # P1 = 0.5*(a+b+2) z + 0.5*(a-b)
            P[..., 1] = 0.5 * (a + b + 2.0) * z + 0.5 * (a - b)

        # 递推：P_n = (A_n z + B_n) P_{n-1} + C_n P_{n-2}
        for n in range(2, K):
            An = ((2*n + a + b - 1) * (2*n + a + b)) / (2*n * (n + a + b))
            Bn = ((2*n + a + b - 1) * (a*a - b*b)) / (2*n * (n + a + b) * (2*n + a + b - 2))
            Cn = (-2.0 * (n + a - 1) * (n + b - 1) * (2*n + a + b)) / (2*n * (n + a + b) * (2*n + a + b - 2))
            # 确保中间计算连续，避免视图导致的 as_strided 反向问题
            Pn_1 = P[..., n-1].contiguous()
            Pn_2 = P[..., n-2].contiguous()
            P[..., n] = (An * z + Bn) * Pn_1 + Cn * Pn_2

        return P.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        利用 Jacobi 基最小二乘拟合系数:
        x: (B, in_features)
        y: (B, in_features, out_features)
        return: (out_features, in_features, degree+1)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        z = self._scale_input(x).contiguous()              # (B, in)
        A = self.jacobi_basis(z).contiguous()              # (B, in, K)
        Bsz, Fin, K = A.shape
        coeff = x.new_zeros(self.out_features, self.in_features, K)

        for i in range(self.in_features):
            Ai = A[:, i, :].contiguous()   # (B, K)
            Bi = y[:, i, :].contiguous()   # (B, out)
            Ci = torch.linalg.lstsq(Ai, Bi).solution  # (K, out)
            coeff[:, i, :] = Ci.T  # (out, K)

        return coeff.contiguous()

    @property
    def scaled_kernel_weight(self):
        if self.enable_standalone_scale_kernel:
            return (self.kernel_weight * self.kernel_scaler.unsqueeze(-1)).contiguous()
        else:
            return self.kernel_weight.contiguous()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Base 分支（非原地激活）
        base_out = F.linear(self.base_activation(x), self.base_weight)  # (B, out)

        # Jacobi 分支
        z = self._scale_input(x).contiguous()          # (B, in)
        A = self.jacobi_basis(z).contiguous()          # (B, in, K)
        A_flat = A.reshape(x.size(0), -1).contiguous() # (B, in*K)
        W_flat = self.scaled_kernel_weight.reshape(self.out_features, -1).contiguous()  # (out, in*K)
        kernel_out = F.linear(A_flat, W_flat)          # (B, out)

        return base_out + kernel_out

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        对 Jacobi 基无需自适应网格，保留为空操作以兼容外部调用。
        """
        return

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 对系数做与原实现一致的 L1 + 熵正则
        l1_fake = self.kernel_weight.abs().mean(-1)  # (out, in)
        regularization_loss_activation = l1_fake.sum()
        denom = (regularization_loss_activation + 1e-12)
        p = (l1_fake / denom).clamp_min(1e-12)
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    """
    Stacked KAN using Jacobi polynomial KANLinear layers.
    """
    def __init__(
        self,
        layers_hidden,
        grid_size=5,          # Jacobi degree
        scale_noise=0.1,
        scale_base=1.0,
        scale_kernel=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,        # 兼容
        grid_range=[-1, 1],
        alpha=0.0,            # Jacobi alpha
        beta=0.0,             # Jacobi beta
        input_scaling='tanh', # 'tanh' or 'linear'
    ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_kernel=scale_kernel,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    alpha=alpha,
                    beta=beta,
                    input_scaling=input_scaling,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=True):
        # 为兼容外部调用保留该参数；Jacobi 模式不使用网格更新
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
