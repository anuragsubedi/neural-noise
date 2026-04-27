import os
import glob
import shutil
from datetime import datetime
import pretty_midi
import numpy as np
import scipy.io.wavfile as wavfile

def convert_and_import_colab_midis():
    source_dir = "data/generated_midi_from_colab"
    target_dir = "data/generated"
    
    os.makedirs(target_dir, exist_ok=True)
    
    midi_files = glob.glob(os.path.join(source_dir, "*.mid*"))
    
    if not midi_files:
        print(f"No MIDI files found in {source_dir}.")
        return

    print(f"Found {len(midi_files)} MIDI files from Colab runs. Processing...")

    for midi_path in midi_files:
        base_name = os.path.basename(midi_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Standardize name for the target directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_name_base = f"colab_{name_without_ext}_{timestamp}"
        
        out_midi_path = os.path.join(target_dir, f"{target_name_base}.mid")
        out_wav_path = os.path.join(target_dir, f"{target_name_base}.wav")
        
        print(f"-> Processing {base_name}...")
        
        try:
            # 1. Copy the MIDI file to the standardized generated folder
            shutil.copy2(midi_path, out_midi_path)
            
            # 2. Synthesize to WAV
            pm = pretty_midi.PrettyMIDI(out_midi_path)
            audio_data = pm.synthesize(fs=44100)
            
            if len(audio_data) > 0 and np.max(np.abs(audio_data)) > 0:
                audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            else:
                audio_data = np.int16(audio_data)

            wavfile.write(out_wav_path, 44100, audio_data)
            
            print(f"   Saved as {out_midi_path} and {out_wav_path}")
            
        except Exception as e:
            print(f"   Error processing {base_name}: {e}")

if __name__ == "__main__":
    convert_and_import_colab_midis()
    print("Done converting Colab MIDIs.")
