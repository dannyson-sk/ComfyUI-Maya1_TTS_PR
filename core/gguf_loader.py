"""
GGUF model loader for Maya1 TTS.
Loads quantized GGUF models and creates Maya1-compatible model wrappers.
"""

import gguf
import torch
import warnings
from pathlib import Path
from typing import Optional
from .gguf_ops import GGMLTensor, replace_linear_with_ggml


class Maya1GGUFModel:
    """
    Wrapper class for Maya1 GGUF model with tokenizer.
    Compatible with the regular Maya1Model interface.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        quantization_type: str,
        device: str
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.quantization_type = quantization_type
        self.device = device

    def __repr__(self):
        return (f"Maya1GGUFModel(name={self.model_name}, "
                f"quant={self.quantization_type}, "
                f"device={self.device})")


def load_gguf_state_dict(gguf_path: Path) -> tuple:
    """
    Load GGUF file and extract state dict with quantized tensors.

    Args:
        gguf_path: Path to .gguf file

    Returns:
        Tuple of (state_dict, quantization_info)
    """
    print(f"📦 Loading GGUF file: {gguf_path.name}")

    reader = gguf.GGUFReader(str(gguf_path))

    state_dict = {}
    qtype_counts = {}

    # Convert GGUF tensors to PyTorch state dict
    for tensor in reader.tensors:
        tensor_name = tensor.name

        # Convert numpy array to torch tensor (memory-mapped)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)

        # Get original shape (GGUF stores shapes in reverse)
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        # Wrap in GGMLTensor to track quantization info
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Not quantized, just reshape
            torch_tensor = torch_tensor.view(*shape)
            state_dict[tensor_name] = torch_tensor
        else:
            # Quantized - wrap with metadata
            state_dict[tensor_name] = GGMLTensor(
                torch_tensor,
                tensor_type=tensor.tensor_type,
                tensor_shape=shape
            )

        # Track quantization types
        qtype_name = getattr(tensor.tensor_type, "name", str(tensor.tensor_type))
        qtype_counts[qtype_name] = qtype_counts.get(qtype_name, 0) + 1

    print(f"   Loaded {len(state_dict)} tensors")
    print(f"   Quantization types: {', '.join(f'{k} ({v})' for k, v in qtype_counts.items())}")

    # Detect primary quantization type
    if qtype_counts:
        primary_qtype = max(qtype_counts.keys(), key=lambda k: qtype_counts[k])
    else:
        primary_qtype = "F16"

    return state_dict, primary_qtype


def load_tokenizer_for_gguf(gguf_path: Path):
    """
    Load tokenizer for GGUF model.

    Tries multiple approaches:
    1. Load from same directory as GGUF (if tokenizer/ folder exists)
    2. Load from default Maya1 tokenizer in models directory
    3. Use hardcoded tokenizer path

    Args:
        gguf_path: Path to .gguf file

    Returns:
        Loaded tokenizer
    """
    from transformers import AutoTokenizer

    # Try 1: Look for tokenizer in same directory as GGUF
    gguf_dir = gguf_path.parent
    if (gguf_dir / "tokenizer").exists():
        print(f"   Loading tokenizer from: {gguf_dir / 'tokenizer'}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(gguf_dir),
            subfolder="tokenizer",
            trust_remote_code=True
        )
        return tokenizer

    # Try 2: Look for maya1 model directory with tokenizer
    # Navigate up to find models/maya1-TTS/maya1/tokenizer
    try:
        from .utils import get_maya1_models_dir
        models_dir = get_maya1_models_dir()

        # Check common model directories
        for model_dir in ["maya1", "Maya1", "maya-research-maya1"]:
            tokenizer_path = models_dir / model_dir
            if (tokenizer_path / "tokenizer").exists():
                print(f"   Loading tokenizer from: {tokenizer_path / 'tokenizer'}")
                tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_path),
                    subfolder="tokenizer",
                    trust_remote_code=True
                )
                return tokenizer
    except Exception as e:
        print(f"   ⚠️  Could not auto-detect tokenizer: {e}")

    # Try 3: Fallback - download from HuggingFace
    print(f"   ⚠️  Tokenizer not found locally, downloading from HuggingFace...")
    print(f"   This may take a moment on first run.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "maya-research/maya1",
            subfolder="tokenizer",
            trust_remote_code=True
        )
        return tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer!\n\n"
            f"Please ensure you have either:\n"
            f"1. A 'tokenizer/' folder next to your GGUF file, OR\n"
            f"2. A Maya1 model in {models_dir}, OR\n"
            f"3. Internet connection to download from HuggingFace\n\n"
            f"Error: {e}"
        )


def unpermute_qk_weights(tensor, n_head):
    """
    Reverse the Q/K weight permutation done by llama.cpp during GGUF conversion.
    Llama.cpp permutes the weights for its own layout, we need to reverse it.

    Args:
        tensor: Weight tensor to unpermute
        n_head: Number of attention heads

    Returns:
        Unpermuted tensor
    """
    from .gguf_dequant import is_quantized, dequantize_tensor

    # Dequantize if needed for permutation
    was_quantized = is_quantized(tensor)
    if was_quantized:
        original_tensor = tensor
        tensor = dequantize_tensor(tensor, dtype=torch.float32)

    # Unpermute: reverse of llama.cpp's permute operation
    # Original shape: [dim, dim] -> reshape -> swapaxes -> reshape
    shape = tensor.shape
    tensor_reshaped = tensor.reshape(n_head, shape[0] // n_head // 2, 2, *shape[1:])
    tensor_swapped = tensor_reshaped.swapaxes(1, 2)
    tensor_unpermuted = tensor_swapped.reshape(shape)

    # Return original quantized tensor if it was quantized
    # (We don't want to keep dequantized version in memory)
    if was_quantized:
        return original_tensor
    else:
        return tensor_unpermuted


def remap_gguf_keys(state_dict: dict, config=None) -> dict:
    """
    Remap GGUF key names to transformers key names.
    GGUF uses llama.cpp naming (blk.X.attn_q) while transformers uses (model.layers.X.self_attn.q_proj).

    Args:
        state_dict: State dict with GGUF key names
        config: Model config (for head count information)

    Returns:
        State dict with transformers key names
    """
    # Key mapping from GGUF (llama.cpp) to transformers
    key_map = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
        "blk.": "model.layers.",
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_lm_head.weight": "self_attn.o_proj.weight",  # Maya1 specific
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
    }

    remapped = {}
    for key, tensor in state_dict.items():
        # Skip RoPE frequencies (computed dynamically in transformers)
        if "rope_freqs" in key:
            continue

        new_key = key
        # Apply all mappings
        for old_pattern, new_pattern in key_map.items():
            new_key = new_key.replace(old_pattern, new_pattern)

        # Mark linear layer weights for transposition
        # GGUF stores Linear weights as [in_features, out_features]
        # PyTorch expects [out_features, in_features]
        # We'll mark them and handle transpose in the forward pass to keep quantization
        if ".weight" in new_key and hasattr(tensor, 'tensor_shape') and len(tensor.tensor_shape) == 2:
            # Check if this is a Linear layer weight
            if any(linear_key in new_key for linear_key in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                # Mark that this weight needs transposition
                tensor.needs_transpose = True
                # Swap the shape dimensions for correct shape reporting
                tensor.tensor_shape = torch.Size([tensor.tensor_shape[1], tensor.tensor_shape[0]])

        # Unpermute Q and K weights (llama.cpp permutes these)
        # NOTE: Disabled for now - may not be needed for Maya1 or might cause issues
        # if config and ("q_proj.weight" in new_key or "k_proj.weight" in new_key):
        #     n_head = config.num_attention_heads
        #     n_head_kv = getattr(config, "num_key_value_heads", n_head)
        #     heads = n_head if "q_proj" in new_key else n_head_kv
        #     tensor = unpermute_qk_weights(tensor, heads)

        remapped[new_key] = tensor

    # Handle tied embeddings: Maya1 GGUF files don't have separate output.weight
    # The lm_head shares weights with the embedding layer (token_embd)
    if "lm_head.weight" not in remapped and "model.embed_tokens.weight" in remapped:
        print(f"   Using tied embeddings: lm_head.weight = model.embed_tokens.weight")
        remapped["lm_head.weight"] = remapped["model.embed_tokens.weight"]

    print(f"   Remapped {len(remapped)} keys from GGUF to transformers format")
    return remapped


def create_maya1_model_from_gguf(state_dict: dict, device: str = "cuda"):
    """
    Create Maya1 model architecture and load GGUF state dict.

    Args:
        state_dict: State dict with GGUF tensors
        device: Device to load on

    Returns:
        Maya1 model with GGUF weights
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"🔧 Creating Maya1 model architecture...")

    # Create model config
    # We need to infer the config from the state dict or use a default
    try:
        # Try to create config from known Maya1 architecture
        config = AutoConfig.from_pretrained(
            "maya-research/maya1",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"   ⚠️  Could not load config from HuggingFace: {e}")
        raise RuntimeError(
            "Failed to load Maya1 config.\n"
            "Please ensure you have internet connection for first-time setup."
        )

    # Remap GGUF keys to transformers keys
    print(f"   Remapping GGUF keys to transformers format...")
    state_dict = remap_gguf_keys(state_dict, config)

    # Create model architecture (on CPU first, then move weights to target device)
    print(f"   Creating model architecture...")
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True
    )

    # Replace Linear/LayerNorm/RMSNorm layers with GGML versions that handle quantization
    print(f"   Replacing layers with GGUF-compatible operations...")
    model = replace_linear_with_ggml(model)

    # Load GGUF weights using PyTorch's built-in load_state_dict
    print(f"   Loading GGUF weights to {device}...")

    # Move tensors to target device
    state_dict_on_device = {}
    for key, tensor in state_dict.items():
        state_dict_on_device[key] = tensor.to(device)

    # Use PyTorch's load_state_dict with our custom _load_from_state_dict overrides
    incompatible_keys = model.load_state_dict(state_dict_on_device, strict=False)

    missing_keys = incompatible_keys.missing_keys
    unexpected_keys = incompatible_keys.unexpected_keys

    if missing_keys:
        print(f"   ⚠️  Missing keys: {len(missing_keys)}")
        for key in missing_keys[:10]:
            print(f"      - {key}")

    if unexpected_keys:
        print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)}")
        for key in unexpected_keys[:10]:
            print(f"      - {key}")

    # Move entire model to target device (for any remaining components)
    print(f"   Finalizing model on {device}...")
    model = model.to(device)
    model.eval()

    return model


def load_gguf_maya1(
    gguf_path: Path,
    attention_type: str = "sdpa",
    device: str = "cuda"
) -> Maya1GGUFModel:
    """
    Load Maya1 model from GGUF file.

    Args:
        gguf_path: Path to .gguf file
        attention_type: Attention mechanism (sdpa/eager)
        device: Device to load on

    Returns:
        Maya1GGUFModel wrapper
    """
    print("=" * 70)
    print("📦 Loading GGUF Maya1 Model")
    print("=" * 70)
    print(f"File: {gguf_path}")
    print(f"Attention: {attention_type}")
    print(f"Device: {device}")

    # Load GGUF state dict
    state_dict, quant_type = load_gguf_state_dict(gguf_path)

    # Load tokenizer
    print(f"🔤 Loading tokenizer...")
    tokenizer = load_tokenizer_for_gguf(gguf_path)
    print(f"   Tokenizer loaded: {len(tokenizer)} tokens")

    # Create model
    model = create_maya1_model_from_gguf(state_dict, device)

    # Create wrapper
    maya1_model = Maya1GGUFModel(
        model=model,
        tokenizer=tokenizer,
        model_name=gguf_path.stem,
        quantization_type=quant_type,
        device=device
    )

    print("=" * 70)
    print(f"✅ GGUF model loaded successfully!")
    print(f"   Quantization: {quant_type}")
    print(f"   Device: {device}")
    print("=" * 70)

    return maya1_model
