import torch
import os
import math
from src.model import MicroMusicGPT, MicroMusicGPTConfig

def get_batch(train_data, val_data, config, split='train'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,)) # batch_size from config
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, eval_iters=50):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, config, split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        perplexity = math.exp(mean_loss)
        out[split] = {'loss': mean_loss, 'perplexity': perplexity}
    model.train()
    return out

def main():
    # Load separate datasets
    train_dataset_path = 'data/dataset_train.pt'
    val_dataset_path = 'data/dataset_val.pt'
    if not os.path.exists(train_dataset_path) or not os.path.exists(val_dataset_path):
        print("Datasets not found. Run tokenizer build script first.")
        return
        
    train_data = torch.load(train_dataset_path)
    val_data = torch.load(val_dataset_path)
    print(f"Loaded {len(train_data)} train tokens and {len(val_data)} validation tokens.")
    
    config = MicroMusicGPTConfig()
    print(f"Instantiating model on device: {config.device}")
    model = MicroMusicGPT(config).to(config.device)
    
    # Check init loss
    xb, yb = get_batch(train_data, val_data, config, 'train')
    _, initial_loss = model(xb, yb)
    print(f"Initial expected loss: {initial_loss.item():.4f}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    max_iters = 500
    eval_interval = 100
    
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Starting training loop...")
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            metrics = estimate_loss(model, train_data, val_data, config)
            print(f"step {iter}: train cross entropyloss {metrics['train']['loss']:.4f} (perplexity {metrics['train']['perplexity']:.4f}), val loss {metrics['val']['loss']:.4f} (ppl {metrics['val']['perplexity']:.4f})")
            # Serialize checkpoint
            checkpoint_path = f"checkpoints/micromusicgpt_v1_step{iter}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            
        xb, yb = get_batch(train_data, val_data, config, 'train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    final_checkpoint = "checkpoints/micromusicgpt_v1_final.pth"
    torch.save(model.state_dict(), final_checkpoint)
    print(f"Training completed. Final weights saved to {final_checkpoint}")

if __name__ == "__main__":
    main()
