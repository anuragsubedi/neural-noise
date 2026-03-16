import mido
import torch
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

class MIDITokenizer:
    def __init__(self):
        self.bos_token = 356
        self.eos_token = 357
        self.vocab_size = 388
        
    def encode(self, midi_path):
        mid = mido.MidiFile(midi_path)
        tokens = [self.bos_token]
        time_buffer_ms = 0
        
        for msg in mid:
            time_buffer_ms += int(round(msg.time * 1000))
            
            if msg.type in ['note_on', 'note_off']:
                is_note_on = msg.type == 'note_on' and msg.velocity > 0
                is_note_off = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
                
                if is_note_on or is_note_off:
                    # Flush accumulated time to sequence
                    while time_buffer_ms >= 10:
                        chunk_ms = min(time_buffer_ms, 1000)
                        chunk_ms = (chunk_ms // 10) * 10 # round down to chunks of 10
                        token = 255 + (chunk_ms // 10)
                        tokens.append(token)
                        time_buffer_ms -= chunk_ms
                        
                    # Emit note bounds
                    if is_note_on:
                        tokens.append(msg.note) # 0-127
                    else:
                        tokens.append(128 + msg.note) # 128-255
                        
        tokens.append(self.eos_token)
        return tokens
        
    def decode(self, tokens, out_path="demo.mid"):
        # Phase 5 Implementation placeholder
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        time_buffer_ms = 0
        ticks_per_beat = mid.ticks_per_beat
        tempo = 500000 # 120 bpm default
        
        for t in tokens:
            if t == self.bos_token or t == self.eos_token:
                continue
            elif 256 <= t <= 355:
                shift_ms = (t - 255) * 10
                time_buffer_ms += shift_ms
            elif 0 <= t <= 127:
                # Convert time_buffer_ms to ticks
                delta_ticks = int(mido.second2tick(time_buffer_ms / 1000.0, ticks_per_beat, tempo))
                track.append(mido.Message('note_on', note=t, velocity=64, time=delta_ticks))
                time_buffer_ms = 0 # reset after emitting event
            elif 128 <= t <= 255:
                note = t - 128
                delta_ticks = int(mido.second2tick(time_buffer_ms / 1000.0, ticks_per_beat, tempo))
                track.append(mido.Message('note_off', note=note, velocity=0, time=delta_ticks))
                time_buffer_ms = 0
                
        mid.save(out_path)

def build_dataset(raw_dir_path, out_train_path, out_val_path):
    import random
    tokenizer = MIDITokenizer()
    
    paths = list(Path(raw_dir_path).rglob("*.midi")) + list(Path(raw_dir_path).rglob("*.mid"))
    # Shuffle for rigid split
    random.seed(42)
    random.shuffle(paths)
    
    split_idx = int(len(paths) * 0.9)
    train_paths = paths[:split_idx]
    val_paths = paths[split_idx:]
    
    def process_split(split_paths, desc):
        tokens = []
        for p in tqdm(split_paths, desc=desc):
            try:
                enc = tokenizer.encode(str(p))
                tokens.extend(enc)
            except Exception as e:
                print(f"Skipping {p} - Error: {e}")
        return tokens
        
    train_tokens = process_split(train_paths, "Tokenizing Train")
    val_tokens = process_split(val_paths, "Tokenizing Val")
    
    print(f"Total train tokens: {len(train_tokens)}")
    print(f"Total val tokens: {len(val_tokens)}")
    
    torch.save(torch.tensor(train_tokens, dtype=torch.long), out_train_path)
    torch.save(torch.tensor(val_tokens, dtype=torch.long), out_val_path)
    print(f"Saved PyTorch datasets: {out_train_path}, {out_val_path}")

if __name__ == "__main__":
    build_dataset("data/raw_midi", "data/dataset_train.pt", "data/dataset_val.pt")
