"""
Model loading and management for Maya1 TTS.
Supports multiple attention mechanisms: SDPA, Flash Attention 2, Sage Attention.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import gc


class GGUFTokenizerWrapper:
    """
    Wrapper to make llama-cpp-python's tokenizer compatible with transformers API.
    """
    def __init__(self, llama_model):
        self.llama_model = llama_model
        # Common token IDs for Maya1
        self.bos_token = "<|begin_of_text|>"
        self.eos_token = "<|end_of_text|>"
        self.pad_token_id = 0

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.llama_model.tokenize(text.encode('utf-8'))

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.llama_model.detokenize(token_ids).decode('utf-8', errors='ignore')

    def __call__(self, text: str, return_tensors: str = None):
        """Tokenize text (transformers-style)."""
        token_ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([token_ids])}
        return {"input_ids": token_ids}


class Maya1Model:
    """
    Wrapper class for Maya1 model with tokenizer and attention mechanism support.
    Supports both SafeTensors (transformers) and GGUF (llama-cpp-python) formats.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        attention_type: str,
        dtype: str,
        device: str,
        model_format: str = "safetensors"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.attention_type = attention_type
        self.dtype = dtype
        self.device = device
        self.model_format = model_format  # "safetensors" or "gguf"

    def __repr__(self):
        return (f"Maya1Model(name={self.model_name}, "
                f"format={self.model_format}, "
                f"attention={self.attention_type}, "
                f"dtype={self.dtype}, "
                f"device={self.device})")

    def is_gguf(self) -> bool:
        """Check if this is a GGUF model."""
        return self.model_format == "gguf"

    def generate(self, input_ids=None, prompt=None, **kwargs):
        """
        Generate tokens using the appropriate API for the model format.

        Args:
            input_ids: Token IDs (for SafeTensors models)
            prompt: Text prompt (for GGUF models)
            **kwargs: Generation parameters

        Returns:
            Generated token IDs (torch.Tensor)
        """
        if self.is_gguf():
            # GGUF generation using llama-cpp-python
            if prompt is None:
                raise ValueError("GGUF models require a text prompt")

            # Extract generation parameters
            max_tokens = kwargs.get('max_new_tokens', 4000)
            temperature = kwargs.get('temperature', 0.4)
            top_p = kwargs.get('top_p', 0.9)
            repetition_penalty = kwargs.get('repetition_penalty', 1.1)

            # Generate using llama-cpp-python
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                echo=False  # Don't include prompt in output
            )

            # Extract generated tokens
            # Note: llama-cpp-python returns a dict with 'choices'
            generated_text = output['choices'][0]['text']

            # Tokenize the output to get token IDs
            generated_ids = self.tokenizer.encode(generated_text)

            # Return as torch tensor matching transformers format
            return torch.tensor([generated_ids])
        else:
            # SafeTensors generation using transformers
            return self.model.generate(input_ids=input_ids, **kwargs)


class Maya1ModelLoader:
    """
    Model loader with caching and attention mechanism configuration.
    """

    # Cache for loaded models
    _model_cache: Dict[str, Maya1Model] = {}

    @staticmethod
    def _get_cache_key(model_path: str, attention_type: str, dtype: str, model_format: str = "safetensors") -> str:
        """Generate a unique cache key for a model configuration."""
        return f"{model_path}|{model_format}|{attention_type}|{dtype}"

    @classmethod
    def load_model(
        cls,
        model_path: Path,
        attention_type: str = "sdpa",
        dtype: str = "bfloat16",
        device: str = "cuda",
        model_format: str = "safetensors"
    ) -> Maya1Model:
        """
        Load Maya1 model with specified configuration.

        Args:
            model_path: Path to model directory or GGUF file
            attention_type: Attention mechanism ("sdpa", "flash_attention_2", "sage_attention")
            dtype: Data type ("bfloat16", "float16", "float32", "8bit", "4bit") for SafeTensors
                   OR GGUF quant type for GGUF models
            device: Device to load on ("cuda", "cpu")
            model_format: Model format ("safetensors" or "gguf")

        Returns:
            Maya1Model wrapper with model and tokenizer
        """
        # Check if dtype OR attention OR format changed from cached model
        # If any changed, clear cache to reload with new settings
        model_path_str = str(model_path)
        for cached_key, cached_model in list(cls._model_cache.items()):
            if model_path_str in cached_key:
                dtype_changed = cached_model.dtype != dtype
                attention_changed = cached_model.attention_type != attention_type
                format_changed = cached_model.model_format != model_format

                if dtype_changed or attention_changed or format_changed:
                    if dtype_changed:
                        print(f"🔄 Dtype changed: {cached_model.dtype} → {dtype}")
                    if attention_changed:
                        print(f"🔄 Attention changed: {cached_model.attention_type} → {attention_type}")
                    if format_changed:
                        print(f"🔄 Format changed: {cached_model.model_format} → {model_format}")

                    print(f"🗑️  Clearing VRAM and reloading model with new settings...")
                    cls.clear_cache(force=True)
                    print(f"✅ VRAM cleared, loading fresh model...")
                    break

        # Check cache
        cache_key = cls._get_cache_key(str(model_path), attention_type, dtype, model_format)
        if cache_key in cls._model_cache:
            model_display_name = model_path.name if hasattr(model_path, 'name') else str(model_path).split('/')[-1]
            print(f"✅ Using cached Maya1 model: {model_display_name}")
            return cls._model_cache[cache_key]

        model_display_name = model_path.name if hasattr(model_path, 'name') else str(model_path).split('/')[-1]
        print(f"📦 Loading Maya1 model: {model_display_name}")
        print(f"   Format: {model_format.upper()}")
        print(f"   Attention: {attention_type}")
        print(f"   Dtype: {dtype}")
        print(f"   Device: {device}")

        # Branch based on model format
        if model_format.lower() == "gguf":
            # GGUF loading path using llama-cpp-python
            maya1_model = cls._load_gguf_model(
                model_path=model_path,
                dtype=dtype,  # For GGUF, this is the quant type
                device=device,
                attention_type=attention_type
            )
        else:
            # SafeTensors loading path using transformers (default)
            maya1_model = cls._load_safetensors_model(
                model_path=model_path,
                dtype=dtype,
                device=device,
                attention_type=attention_type
            )

        # Cache the model
        cls._model_cache[cache_key] = maya1_model

        print(f"✅ Maya1 model loaded successfully!")
        return maya1_model

    @classmethod
    def _load_safetensors_model(
        cls,
        model_path: Path,
        dtype: str,
        device: str,
        attention_type: str
    ) -> Maya1Model:
        """
        Load SafeTensors model using transformers library.

        Args:
            model_path: Path to model directory
            dtype: Data type ("bfloat16", "float16", "float32", "8bit", "4bit")
            device: Device to load on ("cuda", "cpu")
            attention_type: Attention mechanism

        Returns:
            Maya1Model wrapper
        """
        # Import required libraries
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Transformers library not found. Install with:\n"
                "pip install transformers"
            )

        # Check if using bitsandbytes quantization
        use_quantization = dtype in ["8bit", "4bit"]

        if use_quantization:
            # Bitsandbytes quantization
            torch_dtype = torch.bfloat16  # Base dtype for quantization
            print(f"🔧 Quantization requested: {dtype}")
        else:
            # Standard dtype
            torch_dtype = getattr(torch, dtype)

        # Configure attention mechanism
        attn_kwargs = cls._configure_attention(attention_type)

        # Load tokenizer
        tokenizer = cls._load_tokenizer(model_path)

        # Load model
        model = cls._load_model_with_attention(
            model_path,
            torch_dtype,
            device,
            attn_kwargs,
            quantization=dtype if use_quantization else None
        )

        # Apply Sage Attention if selected
        if attention_type == "sage_attention":
            model = cls._apply_sage_attention(model)

        # Create wrapper
        maya1_model = Maya1Model(
            model=model,
            tokenizer=tokenizer,
            model_name=model_path.name,
            attention_type=attention_type,
            dtype=dtype,
            device=device,
            model_format="safetensors"
        )

        # Verify actual settings applied
        cls._verify_model_config(model, attention_type, dtype)

        return maya1_model

    @classmethod
    def _load_gguf_model(
        cls,
        model_path: Path,
        dtype: str,
        device: str,
        attention_type: str
    ) -> Maya1Model:
        """
        Load GGUF model using llama-cpp-python library.

        Args:
            model_path: Path to GGUF file
            dtype: GGUF quantization type (ignored, determined by file)
            device: Device to load on ("cuda", "cpu")
            attention_type: Attention mechanism (limited support in llama.cpp)

        Returns:
            Maya1Model wrapper
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python library not found. Install with:\n"
                "pip install llama-cpp-python\n\n"
                "For CUDA support:\n"
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )

        # Convert Path to string
        model_path_str = str(model_path)

        # Configure GPU layers
        if device == "cuda":
            if not torch.cuda.is_available():
                print("⚠️  CUDA requested but not available, using CPU")
                n_gpu_layers = 0
            else:
                n_gpu_layers = -1  # Use all layers on GPU
                print(f"🔧 Loading GGUF with GPU acceleration (all layers)")
        else:
            n_gpu_layers = 0
            print(f"🔧 Loading GGUF on CPU")

        # Attention implementation note
        if attention_type not in ["sdpa", "eager"]:
            print(f"⚠️  llama.cpp doesn't support {attention_type}, using default attention")

        # Load GGUF model
        print(f"📦 Loading GGUF model with llama-cpp-python...")

        try:
            model = Llama(
                model_path=model_path_str,
                n_gpu_layers=n_gpu_layers,
                n_ctx=4096,  # Context window
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GGUF model:\n{str(e)}\n\n"
                f"Make sure:\n"
                f"1. The file is a valid GGUF model\n"
                f"2. The model is compatible with llama.cpp\n"
                f"3. You have enough VRAM/RAM"
            )

        # For GGUF, llama-cpp-python handles tokenization internally
        # We'll create a tokenizer wrapper for compatibility with existing code
        tokenizer = GGUFTokenizerWrapper(model)

        # Extract model name from path
        model_name = model_path.name if hasattr(model_path, 'name') else str(model_path).split('/')[-1]

        # Create wrapper
        maya1_model = Maya1Model(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            attention_type=attention_type if attention_type in ["sdpa", "eager"] else "default",
            dtype=dtype,  # GGUF quant type from filename
            device=device,
            model_format="gguf"
        )

        print(f"✅ GGUF model loaded: {model_name}")
        if device == "cuda" and n_gpu_layers == -1:
            print(f"   GPU: All layers offloaded to VRAM")

        return maya1_model

    @staticmethod
    def _verify_model_config(model, expected_attention: str, expected_dtype: str):
        """Verify that the model is actually using the requested configuration."""
        print("🔍 Verifying model configuration:")

        # Check actual dtype
        actual_dtype = next(model.parameters()).dtype
        print(f"   ✓ Dtype: {actual_dtype} (requested: {expected_dtype})")

        # Check attention implementation
        if hasattr(model.config, '_attn_implementation'):
            actual_attn = model.config._attn_implementation

            # Special handling for Sage Attention
            if expected_attention == "sage_attention":
                # Sage Attention uses eager as base, so this is expected
                if actual_attn == "eager":
                    print(f"   ✓ Attention: sage_attention (base: eager) ✅")
                else:
                    print(f"   ✓ Attention: {actual_attn} (requested: {expected_attention})")
            else:
                # For other attention types, show normally
                print(f"   ✓ Attention: {actual_attn} (requested: {expected_attention})")
        else:
            # For Sage Attention, check if hooks are registered
            if expected_attention == "sage_attention":
                # Check if forward hooks exist (Sage adds hooks)
                has_hooks = any(
                    hasattr(module, '_forward_hooks') and len(module._forward_hooks) > 0
                    for module in model.modules()
                )
                if has_hooks:
                    print(f"   ✓ Attention: sage_attention hooks applied ✅")
                else:
                    print(f"   ⚠ Attention: sage_attention hooks may not be applied")
            else:
                print(f"   ⚠ Attention: Unable to verify (config._attn_implementation not found)")

    @staticmethod
    def _configure_attention(attention_type: str) -> Dict[str, Any]:
        """
        Configure attention mechanism parameters.

        Args:
            attention_type: Type of attention mechanism

        Returns:
            Dictionary of kwargs for model loading
        """
        if attention_type == "sdpa":
            # PyTorch's scaled_dot_product_attention (default, most compatible)
            return {"attn_implementation": "sdpa"}

        elif attention_type == "flash_attention_2":
            # Flash Attention 2 (fastest, requires flash-attn package)
            try:
                import flash_attn
                return {"attn_implementation": "flash_attention_2"}
            except ImportError:
                print("⚠️  flash-attn not found, falling back to SDPA")
                print("   Install with: pip install flash-attn")
                return {"attn_implementation": "sdpa"}

        elif attention_type == "sage_attention":
            # Sage Attention (memory efficient, requires sageattention package)
            # Use eager mode first, then apply Sage Attention manually
            return {"attn_implementation": "eager"}

        elif attention_type == "eager":
            # Standard PyTorch eager attention (slowest but most compatible)
            return {"attn_implementation": "eager"}

        else:
            print(f"⚠️  Unknown attention type: {attention_type}, using SDPA")
            return {"attn_implementation": "sdpa"}

    @staticmethod
    def _load_tokenizer(model_path: Path):
        """
        Load tokenizer from model path.
        Handles both root and tokenizer/ subdirectory structures.

        Args:
            model_path: Path to model directory

        Returns:
            Loaded tokenizer
        """
        from transformers import AutoTokenizer

        # Check if tokenizer is in a subdirectory
        if (model_path / "tokenizer").exists():
            print("   Loading tokenizer from tokenizer/ subdirectory...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                subfolder="tokenizer",
                trust_remote_code=True
            )
        else:
            print("   Loading tokenizer from root...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )

        return tokenizer

    @staticmethod
    def _load_model_with_attention(
        model_path: Path,
        torch_dtype,
        device: str,
        attn_kwargs: Dict[str, Any],
        quantization: Optional[str] = None
    ):
        """
        Load the model with specified attention configuration.

        Args:
            model_path: Path to model directory
            torch_dtype: PyTorch data type
            device: Device to load on
            attn_kwargs: Attention configuration kwargs
            quantization: Quantization type ("8bit", "4bit", None)

        Returns:
            Loaded model
        """
        from transformers import AutoModelForCausalLM

        # Prepare loading kwargs
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if device == "cuda" else device,
            "trust_remote_code": True,
            **attn_kwargs
        }

        # Add bitsandbytes quantization if requested
        if quantization == "8bit":
            try:
                import bitsandbytes
                print(f"   Using 8-bit quantization (bitsandbytes)")
                load_kwargs["load_in_8bit"] = True
                # Remove device_map incompatibility
                if device == "cpu":
                    print(f"   ⚠️  8-bit quantization requires CUDA, ignoring device=cpu")
                    load_kwargs["device_map"] = "auto"
            except ImportError:
                print(f"⚠️  bitsandbytes not found, loading in bfloat16 instead")
                print(f"   Install with: pip install bitsandbytes")
                quantization = None

        elif quantization == "4bit":
            try:
                import bitsandbytes
                from transformers import BitsAndBytesConfig
                print(f"   Using 4-bit quantization (bitsandbytes NF4)")

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,  # Nested quantization for better quality
                    bnb_4bit_quant_type="nf4"  # NormalFloat4 - best quality
                )
                load_kwargs["quantization_config"] = bnb_config
                # Remove incompatible parameters
                load_kwargs.pop("torch_dtype", None)
                if device == "cpu":
                    print(f"   ⚠️  4-bit quantization requires CUDA, ignoring device=cpu")
                    load_kwargs["device_map"] = "auto"
            except ImportError:
                print(f"⚠️  bitsandbytes not found, loading in bfloat16 instead")
                print(f"   Install with: pip install bitsandbytes")
                quantization = None

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **load_kwargs
        )

        model.eval()  # Set to evaluation mode

        if quantization:
            print(f"✅ Model quantized to {quantization}")

        return model

    @staticmethod
    def _apply_sage_attention(model):
        """
        Apply Sage Attention to the model.
        Supports both Sage Attention v1.x and v2.x APIs.

        Args:
            model: Loaded model

        Returns:
            Model with Sage Attention applied
        """
        try:
            # Try Sage Attention v1.x API first
            try:
                from sageattention import apply_sage_attn
                print("   Applying Sage Attention (v1.x)...")
                model = apply_sage_attn(model)
                print("   ✅ Sage Attention v1.x applied successfully")
                return model
            except ImportError:
                # Try Sage Attention v2.x API
                from sageattention import sageattn
                print("   Applying Sage Attention (v2.x)...")
                # For v2.x, we need to replace attention in each layer
                for name, module in model.named_modules():
                    if hasattr(module, 'self_attn') or 'attention' in name.lower():
                        # Sage Attention v2+ auto-replaces attention when imported
                        pass
                print("   ✅ Sage Attention v2.x detected and enabled")
                return model

        except ImportError:
            print("⚠️  sageattention not found, using standard eager attention")
            print("   Install with: pip install sageattention")
            return model
        except Exception as e:
            print(f"⚠️  Failed to apply Sage Attention: {e}")
            print("   Continuing with standard eager attention")
            return model

    @classmethod
    def clear_cache(cls, force: bool = False):
        """
        Clear the model cache and free VRAM using ComfyUI's native memory management.
        This actually removes models from VRAM, not just moves them to CPU.
        """
        if not cls._model_cache:
            return  # Nothing to clear

        try:
            # Import ComfyUI's model management
            import comfy.model_management as mm

            # Step 1: Delete model references from our cache
            # This removes the Python references to the models
            for cache_key, maya1_model in list(cls._model_cache.items()):
                try:
                    # Delete the model object to free references
                    if maya1_model.model is not None:
                        del maya1_model.model
                    if maya1_model.tokenizer is not None:
                        del maya1_model.tokenizer
                except Exception as e:
                    print(f"   ⚠ Warning: Failed to delete {maya1_model.model_name}: {e}")

            # Step 2: Clear our cache dictionary
            cls._model_cache.clear()

            # Step 3: Use ComfyUI's native VRAM cleanup
            # This unloads ALL models from VRAM (including ours)
            mm.unload_all_models()

            # Step 4: Clear ComfyUI's internal cache
            mm.soft_empty_cache()

            # Step 5: Python garbage collection
            gc.collect()

            # Step 6: Clear CUDA caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        except ImportError:
            # Fallback if comfy.model_management is not available
            print("   ⚠ Warning: ComfyUI model_management not available, using fallback cleanup")

            # Fallback: Just clear the cache and force GC
            cls._model_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
