import os
import csv
import shutil
from pathlib import Path

def main():
    base_dir = Path("/Users/anuragsubedi/Desktop/codebase/neural-noise")
    maestro_dir = base_dir / "maestro-v3.0.0"
    csv_file = maestro_dir / "maestro-v3.0.0.csv"
    dest_dir = base_dir / "data" / "raw_midi"
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            composer = row.get("canonical_composer", "").lower().strip()
            # In MAESTRO v3, Beethoven might be "ludwig van beethoven" or "l v beethoven" (as noted in instructions)
            if "beethoven" in composer:
                midi_filename = row.get("midi_filename")
                if midi_filename:
                    src_path = maestro_dir / midi_filename
                    if src_path.exists():
                        # Use a flat structure in raw_midi, replacing slashes with dashes to avoid collision if necessary
                        # Or just keep the filename
                        dest_filename = midi_filename.replace("/", "_")
                        dest_path = dest_dir / dest_filename
                        shutil.copy2(src_path, dest_path)
                        count += 1
                    else:
                        print(f"Warning: {src_path} not found.")

    print(f"Successfully copied {count} Beethoven MIDI files to {dest_dir}")

if __name__ == "__main__":
    main()
