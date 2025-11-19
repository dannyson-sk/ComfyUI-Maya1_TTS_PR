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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to handle GGML tensors during state dict loading."""
        weight_key = f"{prefix}weight"
        bias_key = f"{prefix}bias"

        if weight_key in state_dict:
            self.weight = nn.Parameter(state_dict[weight_key], requires_grad=False)
        else:
            missing_keys.append(weight_key)

        if bias_key in state_dict:
            self.bias = nn.Parameter(state_dict[bias_key], requires_grad=False)
        elif self.bias is not None:
            missing_keys.append(bias_key)

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
            # Ensure non-quantized weights match input dtype and device
            weight = self.weight.to(device=input.device, dtype=input.dtype)

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
                # Ensure non-quantized bias matches input dtype and device
                bias = self.bias.to(device=input.device, dtype=input.dtype)

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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to handle GGML tensors during state dict loading."""
        weight_key = f"{prefix}weight"

        if weight_key in state_dict:
            self.weight = nn.Parameter(state_dict[weight_key], requires_grad=False)
        else:
            missing_keys.append(weight_key)

    def forward(self, input):
        """Forward pass with automatic dequantization."""
        if is_quantized(self.weight):
            weight = dequantize_tensor(self.weight, dtype=torch.float32)
            weight = weight.to(input.device)
        else:
            # Embeddings stay in their loaded dtype (typically float16 for dequantized)
            # but ensure they're on the right device
            weight = self.weight.to(device=input.device)

        return torch.nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )


class GGMLLayerNorm(nn.LayerNorm):
    """
    LayerNorm that handles quantized weights.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        # Don't allocate weights yet
        nn.Module.__init__(self)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = None
            self.bias = None
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to handle GGML tensors during state dict loading."""
        weight_key = f"{prefix}weight"
        bias_key = f"{prefix}bias"

        if weight_key in state_dict:
            self.weight = nn.Parameter(state_dict[weight_key], requires_grad=False)
        elif self.elementwise_affine:
            missing_keys.append(weight_key)

        if bias_key in state_dict:
            self.bias = nn.Parameter(state_dict[bias_key], requires_grad=False)

    def forward(self, input):
        """Forward pass with automatic dequantization."""
        if self.weight is None:
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, None, None, self.eps
            )

        if is_quantized(self.weight):
            weight = dequantize_tensor(self.weight, dtype=input.dtype)
            weight = weight.to(input.device)
        else:
            weight = self.weight
            if weight is not None:
                weight = weight.to(device=input.device, dtype=input.dtype)

        bias = None
        if self.bias is not None:
            if is_quantized(self.bias):
                bias = dequantize_tensor(self.bias, dtype=input.dtype)
                bias = bias.to(input.device)
            else:
                bias = self.bias.to(device=input.device, dtype=input.dtype)

        return torch.nn.functional.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps
        )


class GGMLRMSNorm(nn.Module):
    """
    RMSNorm that handles quantized weights.
    Compatible with transformers LlamaRMSNorm.
    """

    def __init__(self, hidden_size, eps=1e-6):
        nn.Module.__init__(self)
        self.weight = None
        self.variance_epsilon = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to handle GGML tensors during state dict loading."""
        weight_key = f"{prefix}weight"

        if weight_key in state_dict:
            self.weight = nn.Parameter(state_dict[weight_key], requires_grad=False)
        else:
            missing_keys.append(weight_key)

    def forward(self, hidden_states):
        """Forward pass with automatic dequantization."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Get weight
        if is_quantized(self.weight):
            weight = dequantize_tensor(self.weight, dtype=input_dtype)
            weight = weight.to(hidden_states.device)
        else:
            # Ensure non-quantized weight matches input dtype and device
            weight = self.weight.to(device=hidden_states.device, dtype=input_dtype)

        return weight * hidden_states.to(input_dtype)


class GGMLOps:
    """
    Operations factory for GGUF models.
    Provides custom layers that handle quantized weights.
    """
    Linear = GGMLLinear
    Conv1d = GGMLConv1d
    Embedding = GGMLEmbedding
    LayerNorm = GGMLLayerNorm
    RMSNorm = GGMLRMSNorm


def replace_linear_with_ggml(module):
    """
    Recursively replace nn.Linear, nn.Embedding, nn.LayerNorm, and RMSNorm layers with GGML versions.

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
        elif isinstance(child, nn.Embedding):
            # Replace with GGML version
            ggml_embedding = GGMLEmbedding(
                child.num_embeddings,
                child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse
            )
            # Copy weights if they exist
            if child.weight is not None:
                ggml_embedding.weight = child.weight
            setattr(module, name, ggml_embedding)
        elif isinstance(child, nn.LayerNorm):
            # Replace with GGML version
            ggml_layernorm = GGMLLayerNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine
            )
            # Copy weights if they exist
            if hasattr(child, 'weight') and child.weight is not None:
                ggml_layernorm.weight = child.weight
            if hasattr(child, 'bias') and child.bias is not None:
                ggml_layernorm.bias = child.bias
            setattr(module, name, ggml_layernorm)
        elif child.__class__.__name__ == 'LlamaRMSNorm':
            # Replace transformers' LlamaRMSNorm with GGML version
            ggml_rmsnorm = GGMLRMSNorm(
                hidden_size=child.weight.shape[0] if child.weight is not None else 4096,
                eps=getattr(child, 'variance_epsilon', 1e-6)
            )
            # Copy weights if they exist
            if hasattr(child, 'weight') and child.weight is not None:
                ggml_rmsnorm.weight = child.weight
            setattr(module, name, ggml_rmsnorm)
        else:
            # Recursively process children
            replace_linear_with_ggml(child)

    return module
