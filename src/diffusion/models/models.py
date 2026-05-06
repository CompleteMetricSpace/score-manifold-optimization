#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active model primitives for registry-backed architectures.

Only classes required by models registered in registry.py (plus their transitive
runtime helpers) are kept here.
"""

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import einops
import math
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn


class TimeEmbedder(ABC):
    """Minimal time-embedder interface for trace-model type hints."""

    @abstractmethod
    def embed(self, t: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def get_time_emb_dim(self):
        raise NotImplementedError


# ============================================================================
# Active matrix model components used by TraceMLPScoreModel
# ============================================================================

class TraceProbe(nn.Module):
    def __init__(self, d: int, m: int):
        """
        Args:
            d (int): dimension of each square input matrix X (so X ∈ ℝ^{d×d}).
            m (int): number of probes. For each i=1..m we have two trainable (d×d) matrices K_i and Q_i.
        """
        super().__init__()
        self.d = d
        self.m = m

        # Create trainable parameters K and Q, each of shape (m, d, d)
        # Initialize them (e.g. with small random values)
        self.K = nn.Parameter(
            torch.randn(m, d, d) * (1.0 / d**0.5)
        )
        self.Q = nn.Parameter(
            torch.randn(m, d, d) * (1.0 / d**0.5)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: a batch of square matrices, shape (B, d, d).

        Returns:
            r: a tensor of shape (B, m), where
               r[b, i] = trace(K[i] @ X[b].T @ Q[i] @ X[b]).
        """
        B, d1, d2 = X.shape
        assert d1 == self.d and d2 == self.d, "Input must be shape (B, d, d)"

        # 1) Prepare X and X^T for “broadcasting” across the m probes.
        #    After unsqueezing, X_expand has shape (m, B, d, d).
        X_expanded       = X.unsqueeze(0).expand(self.m, -1, -1, -1)        # (m, B, d, d)
        X_transpose_exp  = X.transpose(-2, -1).unsqueeze(0).expand(self.m, -1, -1, -1)  # (m, B, d, d)

        # 2) Expand Q and K so that they also broadcast over the B dimension:
        #    Q_expand, K_expand each become (m, B, d, d)
        Q_expand = self.Q.unsqueeze(1).expand(self.m, B, self.d, self.d)
        K_expand = self.K.unsqueeze(1).expand(self.m, B, self.d, self.d)

        # 3) Compute Q_i @ X_b for all i,b:
        #    → shape (m, B, d, d)
        Q_times_X = torch.matmul(Q_expand, X_expanded)

        # 4) Compute (X_b^T) @ (Q_i @ X_b) for all i,b:
        #    → shape (m, B, d, d)
        M = torch.matmul(X_transpose_exp, Q_times_X)

        # 5) Now compute K_i @ M_{i,b} for all i,b:
        #    → shape (m, B, d, d)
        K_times_M = torch.matmul(K_expand, M)

        # 6) Finally, take trace over the last two dims.  torch.diagonal(...).sum(-1) yields (m, B)
        #    Then we transpose to get (B, m).
        trace_vals = torch.diagonal(K_times_M, dim1=-2, dim2=-1).sum(-1)  # → (m, B)
        r = trace_vals.transpose(0, 1)                                   # → (B, m)

        return r
class TraceMLPTimeModuleTimeEmbedding(nn.Module):
    def __init__(self, d: int, m: int, hidden_dim: int, num_layers: int, time_emb: TimeEmbedder, activation="relu"):
        """
        A module that
          1) Applies the TraceProbe to each input matrix X ∈ ℝ^{d×d} to get a query vector r ∈ ℝ^m.
          2) Splits X into its d column‐vectors x₁,...,x_d (each ∈ ℝ^d).
          3) Takes an additional time scalar t ∈ ℝ for each example.
          4) For each column x_i, concatenates [ r; x_i; t ] to form a vector in ℝ^{m + d + 1},
             then runs it through an MLP (shared across columns) whose output dimension is d.
          5) Reassembles the d outputs y₁,...,y_d into a matrix Y ∈ ℝ^{d×d} (columns y_i).

        Args:
            d (int): Dimension of each square input matrix (X ∈ ℝ^{d×d}).
            m (int): Number of trace probes. TraceProbe produces r ∈ ℝ^m.
            hidden_dim (int): Hidden‐layer size in the MLP.
            num_layers (int): Number of layers in the MLP. If num_layers=1, the MLP is a single
                              Linear(m + d + 1 → d). If >1, there are (num_layers−1) hidden layers
                              of size hidden_dim each, with ReLU, and a final Linear(hidden_dim → d).
        """
        super().__init__()
        self.d = d
        self.m = m
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_emb = time_emb

        if activation is None or activation == "relu":
            self.activ = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activ = nn.GELU()
        elif activation == "silu":
            self.activ = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Activation unknown: {activation}")

        # 1) The trace‐probe submodule (produces r ∈ ℝ^m from X ∈ ℝ^{d×d})
        self.probe = TraceProbe(d=d, m=m)

        # 2) Build the MLP that maps ℝ^{(m + d + 1)} → ℝ^d
        layers = []
        input_dim = m + d + self.time_emb.get_time_emb_dim()
        if num_layers == 1:
            # Single‐layer MLP: directly map (m + d + 1) → d
            layers.append(nn.Linear(input_dim, d))
        else:
            # First layer: (m + d + 1) → hidden_dim
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activ)
            # Middle hidden layers: (hidden_dim → hidden_dim)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(self.activ)
            # Final layer: hidden_dim → d
            layers.append(nn.Linear(hidden_dim, d))

        self.mlp = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Tensor of shape (B, d, d), a batch of square matrices.
            t: Tensor of shape (B,), a time scalar for each example.

        Returns:
            Y: Tensor of shape (B, d, d), where each Y[b] is formed by
               feeding [ r[b]; x_i[b]; t[b] ] into the shared MLP to produce column y_i.
        """
        B, d1, d2 = X.shape
        assert d1 == self.d and d2 == self.d, f"Expected X of shape (B, {self.d}, {self.d}), got {X.shape}"
        assert t.shape == (B,), f"Expected t of shape (B,), got {t.shape}"

        # 1) Compute r ∈ ℝ^{B×m} via the TraceProbe
        r = self.probe(X)  # shape = (B, m)

        # 2) Extract columns x_i of each X. After transpose, X_cols[b, i, :] is the i-th column of X[b].
        X_cols = X.transpose(1, 2)  # shape = (B, d, d)

        # 3) Expand r so it can be concatenated with each of the d columns
        #    r_expanded: shape = (B, d, m), where each “slice” [:, i, :] is r[batch]
        r_expanded = r.unsqueeze(1).expand(-1, self.d, -1)  # (B, d, m)

        # 4) Expand t so it can be concatenated; treat t[b] as a scalar feature
        #    t_cols: shape = (B, d, 1), where every column in a single example sees the same t[b]
        t_emb = self.time_emb.embed(t)  # provided elsewhere
        #assert t_emb.dim() == 2 and t_emb.shape[0] == B, f"Expected t_emb of shape (B, d_emb), got {t_emb.shape}"
        t_emb_cols = t_emb.unsqueeze(1).expand(-1, self.d, -1)  # (B, d, d_emb)
    
        # 5) Concatenate [r; X_cols; t_emb] along the last dimension → ℝ^{m + d + d_emb}
        #    cat_input[b, i, :] = [ r[b],  X_cols[b, i, :],  t_emb[b] ]
        cat_input = torch.cat((r_expanded, X_cols, t_emb_cols), dim=2)  # (B, d, m + d + d_emb)
    
        # 6) Flatten batch and column‐index dims so we can run the MLP in one go:
        #    flat_input: (B*d, m + d + d_emb)
        flat_input = cat_input.reshape(B * self.d, cat_input.shape[-1])
    
        # 7) Pass through the MLP → (B*d, d)
        flat_output = self.mlp(flat_input)  # shape = (B*d, d)
    
        # 8) Reshape back to (B, d, d), where “d” is the number of columns,
        #    and each output vector is a column y_i.  So output_cols[b, i, :] = y_i
        output_cols = flat_output.view(B, self.d, self.d)  # (B, d, d)
    
        # 9) Transpose columns‐axis back into the second dimension so Y[b, :, i] = y_i
        Y = output_cols.transpose(1, 2)  # (B, d, d)
    
        return Y

# ============================================================================
# Active sequence model components used by UNet1DTimeScoreModel
# ============================================================================

def _gn_groups(c: int) -> int:
    """Choose a sensible GroupNorm group count that divides c."""
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def _center_trim_or_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    If x length != target_len, center-crop or symmetric-pad (zeros) along time dim.
    x: (B, C, L)
    """
    L = x.shape[-1]
    if L == target_len:
        return x
    if L > target_len:  # center-crop
        diff = L - target_len
        start = diff // 2
        end = start + target_len
        return x[..., start:end]
    # pad
    diff = target_len - L
    left = diff // 2
    right = diff - left
    return F.pad(x, (left, right))


# ---------- Time embedding ----------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sin–cos embedding for a scalar time t (B,). Produces (B, time_dim).
    This is similar to diffusion-style timestep embeddings.
    """
    def __init__(self, time_dim: int, max_period: float = 10_000.0, use_2pi: bool = True):
        super().__init__()
        assert time_dim % 2 == 0, "time_dim should be even."
        self.time_dim = time_dim
        self.max_period = max_period
        self.use_2pi = use_2pi

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) arbitrary real values (recommend normalizing to [0,1] or [0,T]).
        """
        device = t.device
        half = self.time_dim // 2
        # Exponential frequency range
        freq_exponents = torch.arange(half, device=device, dtype=t.dtype) / half
        freqs = torch.exp(-math.log(self.max_period) * freq_exponents)  # (half,)
        ang = t[:, None] * freqs[None, :]
        if self.use_2pi:
            ang = ang * (2.0 * math.pi)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, time_dim)
        return emb


class TimeMLP(nn.Module):
    """
    Two-layer MLP used to map sinusoidal embeddings to a richer time context.
    Output is (B, time_mlp_dim).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


# ---------- Core building blocks ----------

class ResBlock1D(nn.Module):
    """
    Residual block with time injection (FiLM-style bias) and GroupNorm.
    - Adds a Linear(t_emb) -> (B, out_channels) then unsqueeze to (B, out_channels, 1)
      and adds to the activations after the first conv.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(_gn_groups(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, L)
        t_emb: (B, time_emb_dim)
        """
        h = self.conv1(self.act1(self.norm1(x)))
        # Inject time
        t_add = self.time_proj(t_emb).unsqueeze(-1)  # (B, C_out, 1)
        h = h + t_add

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample1D(nn.Module):
    """Halve the sequence length with a stride-2 conv; adjust channels."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """
    Double the sequence length via nearest-neighbor + 3x3 conv.
    (Interpolation avoids odd-length artifacts from transposed convs.)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        L = x.shape[-1]
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if target_len is not None:
            x = _center_trim_or_pad(x, target_len)
        return self.conv(x)


# ---------- UNet ----------

class UNet1DTime(nn.Module):
    """
    1D UNet with sinusoidal time embedding. Operates over the sequence axis (N).
    Input  x: (B, N, d)
    Time   t: (B,)
    Output y: (B, N, d)

    Args:
        in_channels:   input feature dim d
        out_channels:  output dim (defaults to in_channels)
        model_channels: base channel width
        channel_mults:  per-level multipliers (depth = len(channel_mults))
        num_res_blocks: # residual blocks per level (encoder & decoder)
        time_emb_dim:   dimension of sin-cos time embedding (even number)
        dropout:        dropout in residual blocks
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        model_channels: int = 64,
        channel_mults: Iterable[int] = (1, 2, 4),# 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

        # Time embedding pipeline
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = TimeMLP(time_emb_dim, model_channels * 4)
        time_mlp_dim = model_channels * 4

        chs: List[int] = [model_channels * m for m in channel_mults]
        depth = len(chs)

        # Initial projection (d -> base C) with 3x3 conv
        self.input_proj = nn.Conv1d(in_channels, chs[0], kernel_size=3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        in_ch = chs[0]
        for i in range(depth):
            out_ch = chs[i]
            stage = nn.ModuleList()
            # Residual stack at this level (keeps channels = out_ch)
            for _ in range(num_res_blocks):
                stage.append(ResBlock1D(in_ch, out_ch, time_mlp_dim, dropout=dropout))
                in_ch = out_ch
            self.down_blocks.append(stage)

            # Downsample to next level (except last)
            if i < depth - 1:
                self.downsamplers.append(Downsample1D(in_ch, chs[i + 1]))
                in_ch = chs[i + 1]

        # Middle (bottleneck)
        mid_ch = chs[-1]
        self.mid_block1 = ResBlock1D(mid_ch, mid_ch, time_mlp_dim, dropout=dropout)
        self.mid_block2 = ResBlock1D(mid_ch, mid_ch, time_mlp_dim, dropout=dropout)

        # Decoder
        self.upsamplers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        in_ch = chs[-1]
        for i in reversed(range(depth)):
            out_ch = chs[i]
            # Upsample from previous deeper features (skip at top level)
            if i < depth - 1:
                # upsample to match the saved skip spatial size; target_len provided in forward
                self.upsamplers.append(Upsample1D(in_ch, out_ch))
                in_ch = out_ch

            # After concat with skip: channels double
            stage = nn.ModuleList()
            # First block sees concatenated channels
            concat_in = in_ch + out_ch if i < depth - 1 else in_ch + out_ch  # consistent handling
            stage.append(ResBlock1D(concat_in, out_ch, time_mlp_dim, dropout=dropout))
            in_ch = out_ch
            # Additional blocks at this level
            for _ in range(num_res_blocks - 1):
                stage.append(ResBlock1D(in_ch, out_ch, time_mlp_dim, dropout=dropout))
            self.up_blocks.append(stage)

        # Final projection back to d with norm + SiLU + 3x3 conv
        self.out_norm = nn.GroupNorm(_gn_groups(chs[0]), chs[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(chs[0], out_channels, kernel_size=3, padding=1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        t: (B,)
        x: (B, N, d)
        returns y: (B, N, d)
        """
        assert x.dim() == 3, "x must be (B, N, d)"
        B, N, d = x.shape
        assert d == self.in_channels, f"Expected last dim {self.in_channels}, got {d}"

        # Prepare
        xt = x.permute(0, 2, 1).contiguous()  # (B, d, N)
        t = t.to(dtype=xt.dtype, device=xt.device)
        t_ctx = self.time_mlp(self.time_embed(t))  # (B, time_mlp_dim)

        # Encoder
        h = self.input_proj(xt)  # (B, C0, N)
        skips: List[torch.Tensor] = []
        sizes: List[int] = []

        for i, stage in enumerate(self.down_blocks):
            for block in stage:
                h = block(h, t_ctx)
            # Save skip after stage
            skips.append(h)
            sizes.append(h.shape[-1])

            # Downsample (except last)
            if i < len(self.downsamplers):
                h = self.downsamplers[i](h)

        # Middle
        h = self.mid_block1(h, t_ctx)
        h = self.mid_block2(h, t_ctx)

        # Decoder
        up_idx = 0
        for i, stage in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            target_len = skip.shape[-1]

            # Upsample (except the very first decoder level if shapes already match)
            if i > 0 or h.shape[-1] < target_len:
                h = self.upsamplers[up_idx](h, target_len=target_len)
                up_idx += 1
            else:
                # Ensure spatial sizes align
                h = _center_trim_or_pad(h, target_len)

            # Concatenate skip
            h = torch.cat([h, skip], dim=1)  # (B, C + C_skip, L)

            # Residual blocks at this level
            for j, block in enumerate(stage):
                h = block(h, t_ctx)

        # Output projection
        h = self.out_conv(self.out_act(self.out_norm(h)))  # (B, out_channels, N)
        y = h.permute(0, 2, 1).contiguous()  # (B, N, out_channels)
        return y


# ============================================================================
# Active temporal U-Net components used by TemporalUnetScoreModel
# ============================================================================

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim, #output_dim + input_dim
        cond_dim, #output_dim
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                if not horizon % 2 == 0:
                    raise Exception("horizon must be multiple of 2 (consider reducing dim_mults length)")
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, t, x):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t_emb = self.time_mlp(t)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t_emb)
            x = resnet2(x, t_emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for resnet, resnet2, attn, upsample in self.ups:
            h_pop = h.pop()
            x = torch.cat((x, h_pop), dim=1)
            x = resnet(x, t_emb)
            x = resnet2(x, t_emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)
