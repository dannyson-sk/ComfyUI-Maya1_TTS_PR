"""
Maya1 TTS GGUF Node for ComfyUI.
Uses pre-quantized GGUF models for lower VRAM usage.
"""

import torch
import numpy as np
import random
import gc
from pathlib import Path
from typing import Tuple
import comfy.model_management as mm

from ..core import (
    SNACDecoder,
    format_prompt,
    load_emotions_list,
    crossfade_audio
)

# Import GGUF-specific utilities
from ..core.gguf_loader import load_gguf_maya1


def create_progress_bar(current: int, total: int, width: int = 12, show_numbers: bool = True) -> str:
    """Create a visual progress bar like ComfyUI's native one."""
    if total == 0:
        percent = 0
    else:
        percent = min(current / total, 1.0)

    filled = int(width * percent)
    empty = width - filled

    bar = '█' * filled + '░' * empty

    if show_numbers:
        return f"[{bar}] {current}/{total}"
    else:
        return f"[{bar}]"


class Maya1TTS_GGUF:
    """
    GGUF quantized version of Maya1 TTS.

    Uses pre-quantized GGUF models from mradermacher/maya1-GGUF.
    Offers significantly lower VRAM usage compared to safetensors:
    - Q4_K_M: ~2GB (vs 6GB BF16)
    - Q6_K: ~2.7GB (higher quality)
    - Q8_0: ~3.5GB (near-lossless)

    Note: Quantization level is baked into the GGUF file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        return {
            "required": {
                # GGUF model path (user enters manually)
                "gguf_model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to .gguf model file (e.g., /path/to/maya1-Q4_K_M.gguf)"
                }),

                # Attention mechanism (limited options for GGUF)
                "attention_mechanism": (["sdpa", "eager"], {
                    "default": "sdpa",
                    "tooltip": "SDPA recommended. Flash/Sage attention may not work with GGUF."
                }),

                "device": (["cuda", "cpu"], {
                    "default": "cuda"
                }),

                # Voice and text
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.",
                    "dynamicPrompts": False
                }),

                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! This is Maya1 <laugh> the best open source voice AI model with emotions.",
                    "dynamicPrompts": False
                }),

                # Generation settings
                "keep_model_in_vram": ("BOOLEAN", {
                    "default": True
                }),

                "temperature": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05
                }),

                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05
                }),

                "max_new_tokens": ("INT", {
                    "default": 4000,
                    "min": 100,
                    "max": 16000,
                    "step": 100,
                    "tooltip": "Maximum NEW SNAC tokens to generate per chunk. 4000 tokens ≈ 30-40s audio"
                }),

                "repetition_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05
                }),

                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),

                "chunk_longform": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Split long text into chunks for unlimited audio length"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/maya1"

    # Cached model to avoid reloading
    _cached_model = None
    _cached_model_path = None

    def generate_speech(
        self,
        gguf_model_path: str,
        attention_mechanism: str,
        device: str,
        voice_description: str,
        text: str,
        keep_model_in_vram: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        repetition_penalty: float,
        seed: int,
        chunk_longform: bool,
    ) -> Tuple[dict]:
        """
        Load GGUF model and generate expressive speech.

        Returns:
            Tuple containing audio dictionary for ComfyUI
        """
        # Check for cancellation
        mm.throw_exception_if_processing_interrupted()

        # Validate GGUF path
        if not gguf_model_path or not gguf_model_path.strip():
            raise ValueError(
                "GGUF model path is required!\n\n"
                "Please enter the path to your .gguf file.\n\n"
                "Download GGUF models from:\n"
                "  https://huggingface.co/mradermacher/maya1-GGUF\n\n"
                "Example:\n"
                "  /path/to/maya1-Q4_K_M.gguf"
            )

        gguf_path = Path(gguf_model_path.strip())

        if not gguf_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {gguf_path}\n\n"
                f"Please check the path and ensure the file exists."
            )

        if not gguf_path.suffix == ".gguf":
            raise ValueError(
                f"File is not a GGUF model: {gguf_path}\n\n"
                f"Please provide a .gguf file."
            )

        # Handle seed
        if seed == 0:
            actual_seed = random.randint(1, 0xffffffffffffffff)
        else:
            actual_seed = seed

        print("=" * 70)
        print("🎤 Maya1 TTS GGUF Generation")
        print("=" * 70)
        print(f"📦 Model: {gguf_path.name}")
        print(f"🎲 Seed: {actual_seed}")
        print(f"💾 VRAM setting: {'Keep in VRAM' if keep_model_in_vram else 'Offload after generation'}")

        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = "cpu"

        # ========== MODEL LOADING ==========
        # Load or use cached model
        if (self._cached_model is None or
            self._cached_model_path != str(gguf_path)):

            print(f"🔄 Loading GGUF model...")

            # Clear old model if exists
            if self._cached_model is not None:
                del self._cached_model
                self._cached_model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Load new model
            maya1_model = load_gguf_maya1(
                gguf_path=gguf_path,
                attention_type=attention_mechanism,
                device=device
            )

            # Cache it
            self._cached_model = maya1_model
            self._cached_model_path = str(gguf_path)
        else:
            print(f"✅ Using cached GGUF model")
            maya1_model = self._cached_model

        mm.throw_exception_if_processing_interrupted()

        # ========== SPEECH GENERATION ==========
        print(f"Voice: {voice_description[:60]}...")
        print(f"Text: {text[:60]}...")
        print(f"Temperature: {temperature}, Top-p: {top_p}")
        print(f"Max tokens: {max_new_tokens}")
        print("=" * 70)

        # Set seed
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)

        # Format prompt using Maya1's OFFICIAL format
        print("🔤 Formatting prompt with control tokens...")

        # Official Maya1 control token IDs
        SOH_ID = 128259  # Start of Header
        EOH_ID = 128260  # End of Header
        SOA_ID = 128261  # Start of Audio
        CODE_START_TOKEN_ID = 128257  # Start of Speech codes
        TEXT_EOT_ID = 128009  # End of Text

        # Decode control tokens
        soh_token = maya1_model.tokenizer.decode([SOH_ID])
        eoh_token = maya1_model.tokenizer.decode([EOH_ID])
        soa_token = maya1_model.tokenizer.decode([SOA_ID])
        sos_token = maya1_model.tokenizer.decode([CODE_START_TOKEN_ID])
        eot_token = maya1_model.tokenizer.decode([TEXT_EOT_ID])
        bos_token = maya1_model.tokenizer.bos_token

        # Build formatted text
        formatted_text = f'<description="{voice_description}"> {text}'

        # Construct full prompt with all control tokens
        prompt = (
            soh_token + bos_token + formatted_text + eot_token +
            eoh_token + soa_token + sos_token
        )

        # Tokenize input
        inputs = maya1_model.tokenizer(
            prompt,
            return_tensors="pt"
        )
        print(f"📊 Input token count: {inputs['input_ids'].shape[1]}")

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Check for cancellation
        mm.throw_exception_if_processing_interrupted()

        # Generate with progress tracking
        print(f"🎵 Generating speech (max {max_new_tokens} tokens)...")

        try:
            # Setup progress tracking
            from comfy.utils import ProgressBar
            progress_bar = ProgressBar(max_new_tokens)

            # Create stopping criteria for cancellation support
            from transformers import StoppingCriteria, StoppingCriteriaList

            class InterruptionStoppingCriteria(StoppingCriteria):
                """Custom stopping criteria that checks for ComfyUI cancellation."""
                def __init__(self, progress_bar):
                    self.progress_bar = progress_bar
                    self.current_tokens = 0
                    self.input_length = 0
                    self.start_time = None

                def __call__(self, input_ids, scores, **kwargs):
                    import time

                    # Store input length on first call
                    if self.input_length == 0:
                        self.input_length = input_ids.shape[1]
                        self.start_time = time.time()

                    # Update progress
                    new_tokens = input_ids.shape[1] - self.input_length
                    if new_tokens > self.current_tokens:
                        self.progress_bar.update(new_tokens - self.current_tokens)
                        self.current_tokens = new_tokens

                        # Print progress
                        if self.current_tokens % 100 == 0:
                            elapsed = time.time() - self.start_time
                            it_per_sec = new_tokens / elapsed if elapsed > 0 else 0
                            print(f"   Generated {new_tokens}/{max_new_tokens} tokens | {it_per_sec:.2f} it/s", end='\r')

                    # Check for cancellation
                    try:
                        mm.throw_exception_if_processing_interrupted()
                    except:
                        print("\n🛑 Generation cancelled by user")
                        return True

                    return False

            stopping_criteria = StoppingCriteriaList([
                InterruptionStoppingCriteria(progress_bar)
            ])

            # Generate tokens
            import time
            generation_start = time.time()

            with torch.inference_mode():
                outputs = maya1_model.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=28,  # At least 4 SNAC frames
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=maya1_model.tokenizer.pad_token_id,
                    eos_token_id=128258,  # SNAC completion token
                    stopping_criteria=stopping_criteria,
                    use_cache=True,
                )

            generation_time = time.time() - generation_start

            # Check for cancellation
            mm.throw_exception_if_processing_interrupted()

            # Extract generated tokens
            generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()

            # Print statistics
            final_speed = len(generated_ids) / generation_time if generation_time > 0 else 0
            print(f"\n✅ Generated {len(generated_ids)} tokens in {generation_time:.2f}s ({final_speed:.2f} it/s)")

            # Filter SNAC tokens
            from ..core.snac_decoder import filter_snac_tokens
            snac_tokens = filter_snac_tokens(generated_ids)

            if len(snac_tokens) == 0:
                raise ValueError(
                    "No SNAC audio tokens generated!\n"
                    "The model may have only generated text tokens.\n"
                    "Try adjusting the prompt or generation parameters."
                )

            print(f"🎵 Found {len(snac_tokens)} SNAC tokens ({len(snac_tokens) // 7} frames)")

            # Check for cancellation
            mm.throw_exception_if_processing_interrupted()

            # Decode SNAC tokens to audio
            print("🔊 Decoding to audio...")
            audio_waveform = SNACDecoder.decode(snac_tokens, device=device)

            # Convert to ComfyUI audio format
            audio_tensor = torch.from_numpy(audio_waveform).float()

            # Add batch and channel dimensions
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)

            audio_output = {
                "waveform": audio_tensor,
                "sample_rate": 24000
            }

            print(f"✅ Generated {len(audio_waveform) / 24000:.2f}s of audio")
            print("=" * 70)

            # Handle VRAM management
            if not keep_model_in_vram:
                print("🗑️  Offloading model from VRAM...")
                del self._cached_model
                self._cached_model = None
                self._cached_model_path = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("✅ Model offloaded from VRAM")
            else:
                print("💾 Model kept in VRAM for faster next generation")

            return (audio_output,)

        except InterruptedError as e:
            print(f"\n{str(e)}")
            print("=" * 70)
            raise

        except Exception as e:
            print(f"\n❌ Generation failed: {str(e)}")
            print("=" * 70)
            raise


# ComfyUI node mappings
NODE_CLASS_MAPPINGS = {
    "Maya1TTS_GGUF": Maya1TTS_GGUF
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Maya1TTS_GGUF": "Maya1 TTS (GGUF)"
}
