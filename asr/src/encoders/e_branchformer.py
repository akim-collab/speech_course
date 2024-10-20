from typing import Optional, Union
from omegaconf import DictConfig
import torch
import torch.nn as nn

from src.submodules.subsampling import StackingSubsampling
from src.submodules.positional_encoding import PositionalEncoding


class ConvolutionalSpatialGatingUnit(nn.Module):
    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_linear_after_conv: bool = False,
    ):
        """
        Convolutional Spatial Gating Unit (https://arxiv.org/pdf/2207.02971)
        Args:
            size: int - Input embedding dim
            kernel_size: int - Kernel size in DepthWise Conv
            dropout: float - Dropout rate
            use_linear_after_conv: bool - Whether to use linear layer after convolution
        """
        super().__init__()
        self.norm = nn.LayerNorm(size // 2)
        self.conv = nn.Conv1d(
            in_channels=size // 2,
            out_channels=size // 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=size // 2,
        )

        if use_linear_after_conv:
            self.linear = nn.Linear(size // 2, size // 2)
        else:
            self.linear = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Inputs:
            x: B x T x C
        Outputs:
            out: B x T x C
        """
        x1, x2 = x.chunk(2, dim=-1)
        x2 = self.conv(self.norm(x2).transpose(1, 2)).transpose(1, 2)
        x = x2 * x1

        if self.linear:
            x = self.linear(x)

        x = self.dropout(x)
        return x


class ConvolutionalGatingMLP(nn.Module):
    def __init__(
        self,
        size: int,
        kernel_size: int,
        expansion_factor: int = 6,
        dropout: float = 0.0,
        use_linear_after_conv: bool = False,
    ):
        """
        Convolutional Gating MLP (https://arxiv.org/pdf/2207.02971)
        Args:
            size: int - Input embedding dim
            kernel_size: int - Kernel size for DepthWise Conv in ConvolutionalSpatialGatingUnit
            expansion_factor: int - Dim expansion factor for ConvolutionalSpatialGatingUnit
            dropout: float - Dropout rate
            use_linear_after_conv: bool - Whether to use linear layer after convolution
        """
        super().__init__()

        hidden_dim = size * expansion_factor

        self.channel_proj1 = nn.Sequential(
            nn.Linear(size, hidden_dim),
            nn.GELU(),
        )

        self.csgu = ConvolutionalSpatialGatingUnit(
            size=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            use_linear_after_conv=use_linear_after_conv,
        )

        self.channel_proj2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, size)
        )

    def forward(self, features: torch.Tensor):
        """
        Inputs:
            features: B x T x C
        Outputs:
            out: B x T x C
        """
        x = self.channel_proj1(features)
        x = self.csgu(x)
        x = self.channel_proj2(x)
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.SiLU(),
    ):
        """
        Standard FeedForward layer from Transformer block,
        consisting of a compression and decompression projection
        with an activation function.
        Args:
            input_dim: int - Input embedding dim
            hidden_dim: int - Hidden dim
            dropout: float - Dropout rate
            activation: nn.Module - Activation function
        """
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, features: torch.Tensor):
        """
        Inputs:
            features: B x T x C
        Outputs:
            out: B x T x C
        """
        x = self.linear1(features)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class EBranchformerEncoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        attn_config: Union[DictConfig, dict],
        cgmlp_config: Union[DictConfig, dict],
        ffn_expansion_factor: int,
        dropout: float,
        merge_conv_kernel: int,
    ):
        """
        E-Bbranchformer Layer (https://arxiv.org/pdf/2210.00077)
        Args:
            size: int - Embedding dim
            attn_config: DictConfig or dict - Config for MultiheadAttention
            cgmlp_config: DictConfig or dict - Config for ConvolutionalGatingMLP
            ffn_expansion_factor: int - Expansion factor for FeedForward
            dropout: float - Dropout rate
            merge_conv_kernel: int - Kernel size for merging module
        """
        super().__init__()

        # MultiheadAttention from torch.nn
        self.attn = nn.MultiheadAttention(**attn_config)
        # ConvolutionalGatingMLP module
        self.cgmlp = ConvolutionalGatingMLP(**cgmlp_config)

        hidden_dim = size * ffn_expansion_factor

        # First and Second FeedForward modules
        self.feed_forward1 = FeedForward(
            input_dim=size,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.feed_forward2 = FeedForward(
            input_dim=size,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Normalization modules
        self.norm_ffn1 = nn.LayerNorm(size)
        self.norm_ffn2 = nn.LayerNorm(size)
        self.norm_mha = nn.LayerNorm(size)
        self.norm_mlp = nn.LayerNorm(size)
        self.norm_final = nn.LayerNorm(size)

        self.dropout = nn.Dropout(dropout)

        # DepthWise Convolution and Linear projection for merging module
        self.depthwise_conv_fusion = nn.Conv1d(
            in_channels=2 * size,
            out_channels=2 * size,
            kernel_size=merge_conv_kernel,
            padding=merge_conv_kernel // 2,
            groups=2 * size,
        )
        self.merge_proj = nn.Linear(2 * size, size)

    def forward(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
    ):
        """
        Inputs:
            features: B x T x C
            features_length: B
            pos_emb: B x T x C - Optional positional embeddings
        Outputs:
            out: B x T x C
        """
        x = features

        # FFN + half-resid
        x_ffn1 = x + self.dropout(self.feed_forward1(self.norm_ffn1(x)) / 2)

        # Global
        x_mha_norm = self.norm_mha(x_ffn1)

        # Create padding mask
        batch_size, seq_len, _ = x_ffn1.size()
        padding_mask = torch.arange(seq_len, device=features_length.device).expand(
            batch_size,
            seq_len
        ) >= features_length.unsqueeze(1)

        attn_output, _ = self.attn(
            query=x_mha_norm,
            key=x_mha_norm,
            value=x_mha_norm,
            need_weights=False,
            key_padding_mask=padding_mask.transpose(0, 1)
        )
        x_global_output = self.dropout(attn_output)

        # Local
        x_local_output = self.norm_mlp(self.cgmlp(x_ffn1))

        # Merge
        x_merged = torch.cat([x_global_output, x_local_output], dim=-1)
        x_conv = self.depthwise_conv_fusion(x_merged.transpose(1, 2)).transpose(1, 2)
        x_merged = self.dropout(self.merge_proj(x_merged + x_conv))
        x_merged = x_ffn1 + x_merged

        # FFN + half-resid + LN
        x_out = self.norm_final(x_merged + self.feed_forward2(self.norm_ffn2(x)) / 2)

        return x_out


class EBranchformerEncoder(nn.Module):
    def __init__(
        self,
        subsampling_stride: int,
        features_num: int,
        d_model: int,
        layers_num: int,
        attn_config: Union[DictConfig, dict],
        cgmlp_config: Union[DictConfig, dict],
        ffn_expansion_factor: int = 4,
        dropout: float = 0.0,
        merge_conv_kernel: int = 31,
    ):
        super().__init__()
        self.subsampling = StackingSubsampling(
            stride=subsampling_stride, feat_in=features_num, feat_out=d_model
        )
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList()
        for _ in range(layers_num):
            layer = EBranchformerEncoderLayer(
                size=d_model,
                attn_config=attn_config,
                cgmlp_config=cgmlp_config,
                ffn_expansion_factor=ffn_expansion_factor,
                dropout=dropout,
                merge_conv_kernel=merge_conv_kernel,
            )
            self.layers.append(layer)

    def forward(self, features: torch.Tensor, features_length: torch.Tensor):
        """
        Inputs:
            features: B x D x T
            features_length: B
        Outputs:
            features: B x T x D
            features_length: B
        """
        features = features.transpose(1, 2)  # B x D x T -> B x T x D
        features, features_length = self.subsampling(features, features_length)
        features = self.pos_embedding(features)
        for layer in self.layers:
            features = layer(features, features_length)

        return features, features_length
