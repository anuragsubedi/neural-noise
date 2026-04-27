import torch
import os
from src.model import MicroMusicGPT, MicroMusicGPTConfig
from src.tokenizer import MIDITokenizer

def load_model(checkpoint_path):
    config = MicroMusicGPTConfig()
    model = MicroMusicGPT(config)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: Checkpoint not found. Generating with uninitialized weights.")
    
    model.to(config.device)
    model.eval()
    return model, config

def generate_midi(model, config, out_path="demo.mid", max_tokens=1000, seed_tokens=None):
    tokenizer = MIDITokenizer()
    
    if seed_tokens is None:
        context = torch.tensor([[tokenizer.bos_token]], dtype=torch.long, device=config.device)
    else:
        context = torch.tensor([seed_tokens], dtype=torch.long, device=config.device)
        
    print("Generating sequence...")
    generated_idx = model.generate(context, max_new_tokens=max_tokens)
    tokens = generated_idx[0].tolist()
    
    print(f"Decoding {len(tokens)} tokens to {out_path}...")
    tokenizer.decode(tokens, out_path=out_path)
    return out_path

if __name__ == "__main__":
    ckpt = "checkpoints/micromusicgpt_v1_final.pth"
    model, config = load_model(ckpt)
    generate_midi(model, config, out_path="demo.mid", max_tokens=1000)
