"""
Core modules for Maya1 TTS ComfyUI integration.
"""

from .model_wrapper import Maya1Model, Maya1ModelLoader
from .snac_decoder import SNACDecoder
from .chunking import (
    smart_chunk_text,
    estimate_tokens_for_text,
    should_chunk_text
)
from .utils import (
    discover_maya1_models,
    get_model_path,
    get_maya1_models_dir,
    load_emotions_list,
    format_prompt,
    check_interruption,
    ProgressCallback,
    crossfade_audio
)

# GGUF support
from .gguf_loader import load_gguf_maya1, Maya1GGUFModel
from .gguf_ops import GGMLTensor, GGMLOps
from .gguf_dequant import is_quantized, dequantize_tensor

__all__ = [
    "Maya1Model",
    "Maya1ModelLoader",
    "SNACDecoder",
    "smart_chunk_text",
    "estimate_tokens_for_text",
    "should_chunk_text",
    "discover_maya1_models",
    "get_model_path",
    "get_maya1_models_dir",
    "load_emotions_list",
    "format_prompt",
    "check_interruption",
    "ProgressCallback",
    "crossfade_audio",
    # GGUF
    "load_gguf_maya1",
    "Maya1GGUFModel",
    "GGMLTensor",
    "GGMLOps",
    "is_quantized",
    "dequantize_tensor",
]
