# GGUF Support for Maya1 TTS

This plugin now supports **GGUF quantized models** for significantly lower VRAM usage!

## What is GGUF?

GGUF (GPT-Generated Unified Format) is a quantization format that reduces model size and VRAM usage while maintaining quality. It's the same format used by llama.cpp and Ollama.

## Benefits

- **3x Lower VRAM**: Q4_K uses ~2GB vs ~6GB for BF16
- **Smaller Downloads**: Q4_K is 2GB vs 6GB full model
- **Good Quality**: Minimal quality loss with Q6_K or Q8_0
- **No Complex Setup**: Just download and use!

## VRAM Comparison

| Format | Size | VRAM Usage | Quality |
|--------|------|------------|---------|
| Safetensors (BF16) | 6GB | ~6GB | Reference |
| Safetensors (4bit BNB) | 6GB | ~3GB | Excellent |
| GGUF Q2_K | 1.4GB | ~1.5GB | Good |
| GGUF Q4_K_M | 2GB | ~2GB | Very Good ⭐ |
| GGUF Q6_K | 2.7GB | ~2.8GB | Excellent |
| GGUF Q8_0 | 3.5GB | ~3.6GB | Near-lossless |

## Installation

1. **Install GGUF dependency:**
   ```bash
   pip install gguf>=0.13.0
   ```

2. **Download a GGUF model:**
   ```bash
   # Q4_K_M - Recommended (2GB, good quality)
   huggingface-cli download mradermacher/maya1-GGUF \
     maya1-Q4_K_M.gguf \
     --local-dir ~/Downloads/

   # Q6_K - Higher quality (2.7GB)
   huggingface-cli download mradermacher/maya1-GGUF \
     maya1-Q6_K.gguf \
     --local-dir ~/Downloads/

   # IQ4_XS - Imatrix quant, best quality for size (1.9GB)
   huggingface-cli download mradermacher/maya1-i1-GGUF \
     maya1.i1-IQ4_XS.gguf \
     --local-dir ~/Downloads/
   ```

3. **Optional: Download tokenizer** (auto-downloaded if not found)
   ```bash
   huggingface-cli download maya-research/maya1 \
     tokenizer/* \
     --local-dir ~/Downloads/maya1-tokenizer/
   ```

## Usage

### In ComfyUI

1. Add **"Maya1 TTS (GGUF)"** node to your workflow
2. Set **gguf_model_path** to your .gguf file:
   ```
   /home/user/Downloads/maya1-Q4_K_M.gguf
   ```
3. Configure voice description and text as normal
4. Generate!

### Example Workflow

```
[Maya1 TTS (GGUF)]
├─ gguf_model_path: "/path/to/maya1-Q4_K_M.gguf"
├─ voice_description: "Realistic female voice..."
├─ text: "Hello world! <laugh>"
├─ temperature: 0.4
└─ [Output] → [Preview Audio] / [Save Audio]
```

## Available Models

All models available at: https://huggingface.co/mradermacher/maya1-GGUF

### Standard Quants
- **Q2_K** (1.44GB) - Smallest, lower quality
- **Q3_K_M** (1.76GB) - Good balance
- **Q4_K_M** (2.09GB) - **Recommended** ⭐
- **Q5_K_M** (2.40GB) - Very good quality
- **Q6_K** (2.72GB) - Excellent quality
- **Q8_0** (3.52GB) - Near-lossless
- **F16** (6.61GB) - Full precision (no quantization)

### Imatrix Quants (Better Quality)
Available at: https://huggingface.co/mradermacher/maya1-i1-GGUF

- **IQ4_XS** (1.91GB) - Best quality for size ⭐⭐
- **Q4_K_M** (2.09GB) - Better than standard Q4_K_M
- All other quant levels with improved quality

## Tokenizer Setup

The GGUF node needs a tokenizer. It will try to find one in this order:

1. **Next to GGUF file**: Place tokenizer in same directory
   ```
   /path/to/
   ├─ maya1-Q4_K_M.gguf
   └─ tokenizer/
      ├─ tokenizer.json
      └─ tokenizer_config.json
   ```

2. **Maya1 models directory**: In `ComfyUI/models/maya1-TTS/maya1/tokenizer/`

3. **Auto-download**: Downloads from HuggingFace (requires internet)

## Limitations

- **Flash Attention / Sage Attention**: May not work with GGUF (use SDPA)
- **Slightly slower**: ~10-20% slower than safetensors due to dequantization
- **Pre-quantized**: Can't change quantization level after download

## Troubleshooting

### Error: "GGUF model path is required"
**Solution**: Enter the full path to your .gguf file

### Error: "Failed to load tokenizer"
**Solution**: Either:
- Place tokenizer folder next to .gguf file, OR
- Download Maya1 safetensors to `models/maya1-TTS/`, OR
- Ensure internet connection for auto-download

### Error: "No SNAC audio tokens generated"
**Solution**:
- Try increasing temperature (0.5-0.7)
- Try a different quantization level (Q6_K or Q8_0)
- Check if voice description is too complex

### Slower than expected
**Solution**:
- Use Q4_K_M or lower (less dequantization overhead)
- Ensure model is on CUDA (not CPU)
- Use SDPA attention (not eager)

## Technical Details

### How it Works

The GGUF implementation:
1. Uses `gguf` Python library (no llama.cpp needed!)
2. Loads quantized weights into memory
3. Dequantizes on-the-fly during forward pass
4. Supports Q2_K through Q8_0 quantization formats

### Why Not Use llama.cpp?

We use pure Python implementation because:
- ✅ No complex wheel dependencies
- ✅ Works across all platforms (Linux, Windows, macOS)
- ✅ No compilation needed
- ✅ Easy to install (`pip install gguf`)
- ✅ Integrates cleanly with PyTorch

### Memory Usage

Quantized weights stay in VRAM but take less space:
- Q4_K: 4 bits per weight (1/8 of FP32)
- Q6_K: 6 bits per weight (3/16 of FP32)
- Q8_0: 8 bits per weight (1/4 of FP32)

Dequantization happens in temporary buffers during forward pass.

## Recommendations

### Best Quality/Size Ratio
**IQ4_XS** from imatrix quants (1.9GB) - Best overall ⭐⭐

### Best for Low VRAM (6-8GB)
**Q4_K_M** (2GB) - Good quality, low VRAM ⭐

### Best Quality
**Q8_0** (3.5GB) or **Q6_K** (2.7GB) - Near-lossless

### Fastest
**Q2_K** (1.4GB) - Smallest model, fastest dequantization

## Credits

GGUF implementation adapted from:
- **ComfyUI-GGUF** by city96 (Apache-2.0)
- **gguf** Python library by ggerganov

GGUF models provided by:
- **mradermacher** on HuggingFace

---

**Need help?** Open an issue at: https://github.com/dannyson-sk/ComfyUI-Maya1_TTS_PR/issues
