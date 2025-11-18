# Adapted from ComfyUI-GGUF (c) City96 || Apache-2.0
"""
Custom PyTorch operations for GGUF quantized weights.
Handles on-the-fly dequantization during forward passes.
"""

import gguf
import torch
import torch.nn as nn
from .gguf_dequant import dequantize_tensor, is_quantized


class GGMLTensor(torch.Tensor):
    """
    Tensor wrapper that stores quantization metadata.
    Weights stay quantized in memory, only dequantized during forward pass.
    """

    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type  # Q4_K, Q6_K, etc.
        self.tensor_shape = tensor_shape

    def __new__(cls, *args, tensor_type, tensor_shape, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        """Override to() to preserve quantization metadata."""
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.needs_transpose = getattr(self, "needs_transpose", False)
        return new

    def clone(self, *args, **kwargs):
        """Override clone() to preserve tensor."""
        return self

    def detach(self, *args, **kwargs):
        """Override detach() to preserve tensor."""
        return self

    @property
    def shape(self):
        """Return original shape before quantization."""
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


class GGMLLinear(nn.Linear):
    """
    Linear layer that dequantizes GGUF weights on-the-fly.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        # Don't allocate memory for weights yet (will be loaded from GGUF)
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = None

    def forward(self, input):
        """
        Forward pass with automatic dequantization.

        If weights are quantized, dequantize them on GPU during forward pass.
        Otherwise, use standard linear operation.
        """
        if is_quantized(self.weight):
            # Dequantize on-the-fly
            weight = dequantize_tensor(self.weight, dtype=input.dtype)
            weight = weight.to(input.device)
        else:
            weight = self.weight

        # Handle GGUF transpose requirement
        # GGUF stores weights as [in, out] but PyTorch expects [out, in]
        if hasattr(self.weight, 'needs_transpose') and self.weight.needs_transpose:
            weight = weight.t()

        # Handle bias
        bias = None
        if self.bias is not None:
            if is_quantized(self.bias):
                bias = dequantize_tensor(self.bias, dtype=input.dtype)
                bias = bias.to(input.device)
            else:
                bias = self.bias

        return torch.nn.functional.linear(input, weight, bias)


class GGMLConv1d(nn.Conv1d):
    """
    Conv1d layer that dequantizes GGUF weights on-the-fly.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = None
        self.bias = None

    def forward(self, input):
        """Forward pass with automatic dequantization."""
        if is_quantized(self.weight):
            weight = dequantize_tensor(self.weight, dtype=input.dtype)
            weight = weight.to(input.device)
        else:
            weight = self.weight

        bias = None
        if self.bias is not None:
            if is_quantized(self.bias):
                bias = dequantize_tensor(self.bias, dtype=input.dtype)
                bias = bias.to(input.device)
            else:
                bias = self.bias

        return self._conv_forward(input, weight, bias)


class GGMLEmbedding(nn.Embedding):
    """
    Embedding layer that dequantizes GGUF weights on-the-fly.
    """

    def forward(self, input):
        """Forward pass with automatic dequantization."""
        if is_quantized(self.weight):
            weight = dequantize_tensor(self.weight, dtype=torch.float32)
            weight = weight.to(input.device)
        else:
            weight = self.weight

        return torch.nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )


class GGMLLayerNorm(nn.LayerNorm):
    """
    LayerNorm that handles quantized weights.
    """

    def forward(self, input):
        """Forward pass with automatic dequantization."""
        if self.weight is None:
            return super().forward(input)

        if is_quantized(self.weight):
            weight = dequantize_tensor(self.weight, dtype=input.dtype)
            weight = weight.to(input.device)
        else:
            weight = self.weight

        bias = None
        if self.bias is not None:
            if is_quantized(self.bias):
                bias = dequantize_tensor(self.bias, dtype=input.dtype)
                bias = bias.to(input.device)
            else:
                bias = self.bias

        return torch.nn.functional.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps
        )


class GGMLOps:
    """
    Operations factory for GGUF models.
    Provides custom layers that handle quantized weights.
    """
    Linear = GGMLLinear
    Conv1d = GGMLConv1d
    Embedding = GGMLEmbedding
    LayerNorm = GGMLLayerNorm


def replace_linear_with_ggml(module):
    """
    Recursively replace nn.Linear layers with GGMLLinear.

    Args:
        module: PyTorch module to modify

    Returns:
        Modified module with GGML operations
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace with GGML version
            ggml_linear = GGMLLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None
            )
            # Copy weights if they exist
            if child.weight is not None:
                ggml_linear.weight = child.weight
            if child.bias is not None:
                ggml_linear.bias = child.bias
            setattr(module, name, ggml_linear)
        else:
            # Recursively process children
            replace_linear_with_ggml(child)

    return module
