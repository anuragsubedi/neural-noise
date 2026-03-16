import os
import sys
import glob

# Add parent directory to path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tokenizer import MIDITokenizer

def create_manual_seeds():
    tokenizer = MIDITokenizer()
    os.makedirs("data/seeds", exist_ok=True)
    
    # --- 1. Basic Block Chords (ii-V-I in C Major) ---
    dmin_on   = [62, 65, 69]
    dmin_off  = [n + 128 for n in dmin_on]
    gmaj_on   = [67, 71, 74]
    gmaj_off  = [n + 128 for n in gmaj_on]
    cmaj_on   = [60, 64, 67]
    cmaj_off  = [n + 128 for n in cmaj_on]
    
    block_seed = [tokenizer.bos_token] + \
                 dmin_on + [305] + dmin_off + \
                 gmaj_on + [305] + gmaj_off + \
                 cmaj_on + [355] + cmaj_off + [tokenizer.eos_token]
                 
    tokenizer.decode(block_seed, out_path="data/seeds/01_manual_block_chords.mid")
    print("Created data/seeds/01_manual_block_chords.mid")

    # --- 2. Rhythmic Progression (ii7-V7-Imaj7) ---
    dmin7_on  = [62, 65, 69, 72]
    dmin7_off = [n + 128 for n in dmin7_on]
    gdom7_on  = [55, 59, 62, 65]
    gdom7_off = [n + 128 for n in gdom7_on]
    cmaj7_on  = [60, 64, 67, 71]
    cmaj7_off = [n + 128 for n in cmaj7_on]
    
    rhythmic_seed = [tokenizer.bos_token] + \
                    dmin7_on + [305] + dmin7_off + \
                    gdom7_on + [305] + gdom7_off + \
                    cmaj7_on + [355] + cmaj7_off + [tokenizer.eos_token]
                    
    tokenizer.decode(rhythmic_seed, out_path="data/seeds/02_manual_rhythmic_chords.mid")
    print("Created data/seeds/02_manual_rhythmic_chords.mid")

def extract_real_excerpts():
    tokenizer = MIDITokenizer()
    raw_files = sorted(glob.glob("data/raw_midi/*.mid*"))
    
    if raw_files:
        import random
        # Grab 3 random files and extract ~3 seconds of human performance
        for i, sample_path in enumerate(random.sample(raw_files, min(len(raw_files), 3))):
            real_tokens = tokenizer.encode(sample_path)
            
            prompt_tokens = [tokenizer.bos_token]
            current_time_ms = 0
            
            for t in real_tokens[1:]:
                if 256 <= t <= 355:
                    shift_ms = (t - 255) * 10
                    current_time_ms += shift_ms
                prompt_tokens.append(t)
                
                # Extract exactly 3 seconds of real human timing
                if current_time_ms >= 3000:
                    break
            
            prompt_tokens.append(tokenizer.eos_token)
            out_name = f"data/seeds/0{3+i}_beethoven_excerpt_{i+1}.mid"
            tokenizer.decode(prompt_tokens, out_path=out_name)
            print(f"Created {out_name} (3-second excerpt from {os.path.basename(sample_path)})")
    else:
        print("No raw midi files found in data/raw_midi/ to create real excerpts.")

if __name__ == "__main__":
    print("Generating seed prompt files for the UI...")
    create_manual_seeds()
    extract_real_excerpts()
    print("\nSuccess! You can now select these from the Gradio Context Dropdown.")
