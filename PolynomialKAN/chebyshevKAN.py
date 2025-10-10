import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    """
    KANLinear with Chebyshev polynomial basis (first kind T_n or second kind U_n).

    - basis: 'chebyshev1' (T_n) or 'chebyshev2' (U_n)
    - grid_size: polynomial degree n  => basis dimension = n+1
    - input_scaling: 'tanh' or 'linear' (per-batch min-max to [-1,1])
    - update_grid: no-op (kept for API compatibility)
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,                 # degree n
        scale_noise=0.1,
        scale_base=1.0,
        scale_kernel=1.0,            # kept for API compatibility
        enable_standalone_scale_kernel=True,  # per-(out,in) scalar on coeffs
        base_activation=torch.nn.SiLU,        # non-inplace
        grid_eps=0.02,               # compat
        grid_range=[-1, 1],          # used if input_scaling='linear' and want fixed range
        input_scaling='tanh',        # 'tanh' or 'linear'
        basis='chebyshev1',          # 'chebyshev1' (T_n) or 'chebyshev2' (U_n)
    ):
        super(KANLinear, self).__init__()
        assert basis in ('chebyshev1', 'chebyshev2')
        self.in_features = in_features
        self.out_features = out_features
        self.degree = int(grid_size)
        self.input_scaling = input_scaling
        self.grid_range = grid_range
        self.basis = basis

        # Base linear branch
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # Polynomial coeffs: (out, in, degree+1)
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
        self.base_activation = base_activation()
        self.grid_eps = grid_eps  # compat

        self.reset_parameters()

    def reset_parameters(self):
        # base linear
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # initialize polynomial coeffs by least squares fit on random small curves in [-1,1]
        with torch.no_grad():
            S = max(2 * (self.degree + 1), 16)
            z = torch.linspace(-1.0, 1.0, S).unsqueeze(1).repeat(1, self.in_features).contiguous()  # (S, in)

            noise = (torch.rand(S, self.in_features, self.out_features) - 0.5) * (self.scale_noise / max(1, self.degree))
            noise = noise.contiguous()

            A = self.chebyshev_basis(z).contiguous()  # (S, in, K)
            K = self.degree + 1
            coeff = torch.zeros(self.out_features, self.in_features, K)

            for i in range(self.in_features):
                Ai = A[:, i, :].contiguous()  # (S, K)
                Bi = noise[:, i, :].contiguous()  # (S, out)
                Ci = torch.linalg.lstsq(Ai, Bi).solution  # (K, out)
                coeff[:, i, :] = Ci.T  # (out, K)

            self.kernel_weight.copy_(coeff)

            if self.enable_standalone_scale_kernel:
                torch.nn.init.kaiming_uniform_(self.kernel_scaler, a=math.sqrt(5) * self.scale_kernel)

    @staticmethod
    def _tanh_scale(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def _linear_scale_batch(x: torch.Tensor) -> torch.Tensor:
        # per-batch, per-feature min-max to [-1,1]
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        denom = (x_max - x_min).clamp_min(1e-6)
        return ((x - x_min) / denom) * 2.0 - 1.0

    def _scale_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scaling == 'tanh':
            return self._tanh_scale(x)
        elif self.input_scaling == 'linear':
            return self._linear_scale_batch(x)
        else:
            raise ValueError("input_scaling must be 'tanh' or 'linear'")

    def chebyshev_basis(self, z: torch.Tensor) -> torch.Tensor:
        """
        Build Chebyshev basis up to degree n for each feature.
        z: (B, in_features) in [-1,1]
        return: (B, in_features, degree+1)
        """
        assert z.dim() == 2
        B, F = z.shape
        K = self.degree + 1
        P = z.new_zeros(B, F, K).contiguous()

        if self.basis == 'chebyshev1':
            # T_0=1, T_1=z, T_n=2z T_{n-1} - T_{n-2}
            P[..., 0] = 1.0
            if self.degree >= 1:
                P[..., 1] = z
            for n in range(2, K):
                Tn_1 = P[..., n-1].contiguous()
                Tn_2 = P[..., n-2].contiguous()
                P[..., n] = 2.0 * z * Tn_1 - Tn_2

        else:
            # chebyshev2: U_0=1, U_1=2z, U_n=2z U_{n-1} - U_{n-2}
            P[..., 0] = 1.0
            if self.degree >= 1:
                P[..., 1] = 2.0 * z
            for n in range(2, K):
                Un_1 = P[..., n-1].contiguous()
                Un_2 = P[..., n-2].contiguous()
                P[..., n] = 2.0 * z * Un_1 - Un_2

        return P.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Fit Chebyshev coefficients via least squares.
        x: (B, in_features)
        y: (B, in_features, out_features)
        returns: (out_features, in_features, degree+1)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        z = self._scale_input(x).contiguous()
        A = self.chebyshev_basis(z).contiguous()
        Bsz, Fin, K = A.shape
        coeff = x.new_zeros(self.out_features, self.in_features, K)

        for i in range(self.in_features):
            Ai = A[:, i, :].contiguous()  # (B, K)
            Bi = y[:, i, :].contiguous()  # (B, out)
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

        # Base branch (non-inplace activation)
        base_out = F.linear(self.base_activation(x), self.base_weight)

        # Chebyshev branch
        z = self._scale_input(x).contiguous()
        A = self.chebyshev_basis(z).contiguous()                  # (B, in, K)
        A_flat = A.reshape(x.size(0), -1).contiguous()            # (B, in*K)
        W_flat = self.scaled_kernel_weight.reshape(self.out_features, -1).contiguous()
        kernel_out = F.linear(A_flat, W_flat)                     # (B, out)

        return base_out + kernel_out

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        # No-op for polynomial basis
        return

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
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
    Stacked KAN using Chebyshev polynomial KANLinear layers.
    """
    def __init__(
        self,
        layers_hidden,
        grid_size=5,          # degree n
        scale_noise=0.1,
        scale_base=1.0,
        scale_kernel=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,        # compat
        grid_range=[-1, 1],
        input_scaling='tanh', # 'tanh' or 'linear'
        basis='chebyshev1',   # 'chebyshev1' or 'chebyshev2'
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
                    input_scaling=input_scaling,
                    basis=basis,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=True):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)  # no-op here
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
