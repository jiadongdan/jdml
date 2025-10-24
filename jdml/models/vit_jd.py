# vit_six_channel.py
# Minimal Vision Transformer for multi-channel images (default: 6 channels)

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------
def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Per-sample stochastic depth (a.k.a. DropPath)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # shape: (batch, 1, 1) to broadcast across sequence/feature dims
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    output = x / keep_prob * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


# ----------------------------
# Core building blocks
# ----------------------------
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using a Conv2d:
      Input : (B, C, H, W)
      Output: (B, N, D) where N = (H/ps)*(W/ps), D = embed_dim
    """
    def __init__(self, img_size: int = 256, patch_size: int = 16,
                 in_chans: int = 6, embed_dim: int = 768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)  # (Gh, Gw)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)            # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-head Self-Attention
      Input/Output: (B, N, D)
    """
    def __init__(self, dim: int, num_heads: int = 12,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, h, N, d)
        q, v = qkv[0], qkv[2]  # each: (B, h, N, d)
        k = qkv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, h, N, d)
        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer Encoder Block
      Input/Output: (B, N, D)
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ----------------------------
# Positional Embeddings (2D sin-cos, no learned params)
# ----------------------------
def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int, cls_token: bool = True) -> torch.Tensor:
    """
    Create 2D sine-cosine positional embeddings.
    Returns: (1, 1+N, D) if cls_token else (1, N, D)
    """
    # Grid of positions
    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=0)  # (2, Gh, Gw)
    grid = grid.reshape(2, 1, grid_h, grid_w)

    # Split half for y, half for x
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos"
    dim_each = embed_dim // 2

    pos_y = _get_1d_sincos_pos_embed_from_grid(dim_each, grid[0])  # (Gh, Gw, D/2)
    pos_x = _get_1d_sincos_pos_embed_from_grid(dim_each, grid[1])  # (Gh, Gw, D/2)

    pos = torch.cat([pos_y, pos_x], dim=-1)  # (Gh, Gw, D)
    pos = pos.reshape(1, grid_h * grid_w, embed_dim)  # (1, N, D)

    if cls_token:
        cls = torch.zeros(1, 1, embed_dim, dtype=pos.dtype)
        pos = torch.cat([cls, pos], dim=1)  # (1, 1+N, D)
    return pos

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    pos: (Gh, Gw) float32 grid
    returns: (Gh, Gw, D)
    """
    # frequencies
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    pos = pos.unsqueeze(-1)  # (Gh, Gw, 1)

    out = torch.cat([torch.sin(pos * omega), torch.cos(pos * omega)], dim=-1)  # (Gh, Gw, D)
    return out


# ----------------------------
# ViT
# ----------------------------
class ViT(nn.Module):
    """
    Vision Transformer for multi-channel images (default: 6 channels).

    Args:
        img_size: assumed training image size (H=W). Inference can use any H,W divisible by patch_size.
        patch_size: patch size (ps). H and W must be divisible by ps.
        in_chans: number of input channels (use 6 for your data).
        num_classes: number of target classes.
        embed_dim: token embedding dimension.
        depth: number of Transformer blocks.
        num_heads: number of attention heads.
        mlp_ratio: expansion ratio in MLP.
        drop: dropout in MLP/projection layers.
        attn_drop: dropout on attention weights.
        drop_path: stochastic depth rate (linearly scaled across depth).

    Shapes:
        Input  : (B, C, H, W)  [e.g., C=6]
        Patches: (B, N, D) where N = (H/ps)*(W/ps)
        Tokens : prepend CLS -> (B, 1+N, D)
        Output : logits (B, num_classes)
    """
    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 16,
            in_chans: int = 6,
            num_classes: int = 7,
            embed_dim: int = 768,
            depth: int = 6,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            qkv_bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 (for 2D sin-cos pos embed)."

        # Patchify
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings are generated on the fly (2D sin-cos), so no learnable pos_embed is stored.

        # Stochastic depth decay rule (linearly increases across layers)
        dpr = torch.linspace(0, drop_path, steps=depth).tolist()

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # Cache for positional embeddings by (Gh, Gw)
        self._pos_cache: dict[Tuple[int, int], torch.Tensor] = {}

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.no_grad()
    def _get_pos_embed(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get/calc positional embedding for a given patch grid size (Gh=W/ps, Gw=H/ps)."""
        Gh = H // self.patch_embed.patch_size
        Gw = W // self.patch_embed.patch_size
        key = (Gh, Gw)
        if key not in self._pos_cache:
            pe = get_2d_sincos_pos_embed(self.patch_embed.proj.out_channels, Gh, Gw, cls_token=True)  # (1, 1+N, D)
            self._pos_cache[key] = pe
        return self._pos_cache[key].to(device=device, dtype=dtype)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, D) pooled CLS token after transformer
        """
        B, C, H, W = x.shape
        assert (H % self.patch_embed.patch_size == 0) and (W % self.patch_embed.patch_size == 0), \
            "H and W must be divisible by patch_size."

        x = self.patch_embed(x)  # (B, N, D)
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1+N, D)

        # Positional embeddings
        pos = self._get_pos_embed(H, W, x.device, x.dtype)  # (1, 1+N, D)
        x = x + pos

        # Transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Take CLS token
        return x[:, 0]  # (B, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 6, H, W) -> logits: (B, num_classes)
        """
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits

