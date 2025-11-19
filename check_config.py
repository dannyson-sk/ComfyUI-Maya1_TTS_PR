from transformers import AutoConfig

config = AutoConfig.from_pretrained('maya-research/maya1', trust_remote_code=True)
print(f'tie_word_embeddings: {config.tie_word_embeddings}')
print(f'vocab_size: {config.vocab_size}')
print(f'hidden_size: {config.hidden_size}')

# Check if lm_head exists in the safetensors model
from transformers import AutoModelForCausalLM
import torch

print("\nChecking safetensors model structure...")
with torch.device('meta'):
    model = AutoModelForCausalLM.from_pretrained('maya-research/maya1', trust_remote_code=True)
    print(f'has lm_head: {hasattr(model, "lm_head")}')
    if hasattr(model, 'lm_head'):
        print(f'lm_head type: {type(model.lm_head)}')
        print(f'lm_head weight shape (expected): {model.lm_head.weight.shape}')
    print(f'embed_tokens shape: {model.model.embed_tokens.weight.shape}')
