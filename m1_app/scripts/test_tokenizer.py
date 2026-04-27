import mido
import os
from pathlib import Path
from src.tokenizer import MIDITokenizer

def test_roundtrip():
    # Find a sample file
    raw_dir = Path("data/raw_midi")
    sample_files = list(raw_dir.glob("*.mid*"))
    if not sample_files:
        print("No raw midi files found.")
        return
        
    sample_file = sample_files[0]
    print(f"Testing roundtrip on: {sample_file.name}")
    
    # 1. Original stats
    mid_orig = mido.MidiFile(sample_file)
    orig_note_on = sum(1 for msg in mid_orig if msg.type == 'note_on' and msg.velocity > 0)
    orig_note_off = sum(1 for msg in mid_orig if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0))
    print(f"Original MIDI -> Note ONs: {orig_note_on}, Note OFFs: {orig_note_off}, Length (s): {mid_orig.length:.2f}")
    
    # 2. Encode
    tokenizer = MIDITokenizer()
    tokens = tokenizer.encode(str(sample_file))
    print(f"Encoded into {len(tokens)} tokens.")
    
    # Analyze tokens
    time_tokens = sum(1 for t in tokens if 256 <= t <= 355)
    print(f"Token Distribution -> Time Shifts: {time_tokens}")
    
    # 3. Decode
    out_path = "test_roundtrip.mid"
    tokenizer.decode(tokens, out_path=out_path)
    
    # 4. Reconstructed stats
    mid_recon = mido.MidiFile(out_path)
    recon_note_on = sum(1 for msg in mid_recon if msg.type == 'note_on' and msg.velocity > 0)
    recon_note_off = sum(1 for msg in mid_recon if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0))
    print(f"Reconstructed MIDI -> Note ONs: {recon_note_on}, Note OFFs: {recon_note_off}, Length (s): {mid_recon.length:.2f}")
    
    if orig_note_on != recon_note_on or orig_note_off != recon_note_off:
        print("FAILED: Mismatched event counts! Tokenizer logic is dropping/hallucinating events.")
    else:
        print("PASSED: Event counts match perfectly.")

if __name__ == "__main__":
    test_roundtrip()
