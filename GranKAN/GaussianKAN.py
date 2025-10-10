import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            scale_noise=0.1,
            scale_base=1.0,
            scale_kernel=1.0,
            enable_standalone_scale_kernel=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(grid_size + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.kernel_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + 1)
        )
        if enable_standalone_scale_kernel:
            self.kernel_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_kernel = scale_kernel
        self.enable_standalone_scale_kernel = enable_standalone_scale_kernel
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.kernel_weight.data.copy_(
                (self.scale_kernel if not self.enable_standalone_scale_kernel else 1.0)
                * self.curve2coeff(
                    self.grid.T,
                    noise,
                )
            )
            if self.enable_standalone_scale_kernel:
                torch.nn.init.kaiming_uniform_(self.kernel_scaler, a=math.sqrt(5) * self.scale_kernel)

    def gaussian_kernel(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        x = x.unsqueeze(-1) - self.grid.unsqueeze(0)
        kernel = torch.exp(-0.5 * (x ** 2) / self.scale_kernel)
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-6)

        assert kernel.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + 1,
        )
        return kernel.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.gaussian_kernel(x).transpose(
            0, 1
        )
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(
            A, B
        ).solution
        result = solution.permute(
            2, 0, 1
        )

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + 1,
        )
        return result.contiguous()

    @property
    def scaled_kernel_weight(self):
        return self.kernel_weight * (
            self.kernel_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_kernel
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        kernel_output = F.linear(
            self.gaussian_kernel(x).view(x.size(0), -1),
            self.scaled_kernel_weight.view(self.out_features, -1),
        )
        return base_output + kernel_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        kernels = self.gaussian_kernel(x)
        kernels = kernels.permute(1, 0, 2)
        orig_coeff = self.scaled_kernel_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_kernel_output = torch.bmm(kernels, orig_coeff)
        unreduced_kernel_output = unreduced_kernel_output.permute(
            1, 0, 2
        )

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        self.grid.copy_(grid.T)
        self.kernel_weight.data.copy_(self.curve2coeff(x, unreduced_kernel_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.kernel_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            scale_noise=0.1,
            scale_base=1.0,
            scale_kernel=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size

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
                )
            )

    def forward(self, x: torch.Tensor, update_grid=True):
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
