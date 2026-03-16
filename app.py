import gradio as gr
import pretty_midi
import matplotlib.pyplot as plt
import torch
import scipy.io.wavfile as wavfile
from generate import load_model, generate_midi
from src.tokenizer import MIDITokenizer
import os
import glob
from datetime import datetime

os.makedirs("data/seeds", exist_ok=True)
os.makedirs("data/generated", exist_ok=True)

model, config = load_model("checkpoints/micromusicgpt_v1_final.pth")
tokenizer = MIDITokenizer()

def get_seed_files():
    return ["None (Empty Start)"] + glob.glob("data/seeds/*.mid*")

def plot_piano_roll(pm, title="Generated Pianoroll Mapping"):
    roll = pm.get_piano_roll(fs=100)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(roll, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def generate_interface(num_tokens, seed_file):
    num_tokens = int(num_tokens)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_midi_path = f"data/generated/demo_{timestamp}.mid"
    out_wav_path = f"data/generated/demo_{timestamp}.wav"
    
    seed_tokens = None
    if seed_file and seed_file != "None (Empty Start)" and os.path.exists(seed_file):
        try:
            # Strip the EOS token so the model knows to continue generating
            full_tokens = tokenizer.encode(seed_file)[:-1]
            # Use context up to 200 tokens to provide a solid musical foundation
            seed_tokens = full_tokens[:200] 
        except Exception as e:
            print(f"Error loading seed: {e}")
            
    try:
        generate_midi(model, config, out_path=out_midi_path, max_tokens=num_tokens, seed_tokens=seed_tokens)
        
        pm = pretty_midi.PrettyMIDI(out_midi_path)
        # Synthesize audio array using pretty_midi
        audio_data = pm.synthesize(fs=44100)
        import numpy as np
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        wavfile.write(out_wav_path, 44100, audio_data)
        
        fig = plot_piano_roll(pm)
        return out_wav_path, fig, out_midi_path
    except Exception as e:
        print(f"Error during sequence UI wrapper: {e}")
        return None, None, None

def get_generated_files():
    return sorted(glob.glob("data/generated/*.mid*"), reverse=True)

def load_generated_sequence(midi_path):
    if not midi_path or not os.path.exists(midi_path):
        return None, None, None
    
    wav_path = os.path.splitext(midi_path)[0] + ".wav"
    if not os.path.exists(wav_path):
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            audio_data = pm.synthesize(fs=44100)
            import numpy as np
            if len(audio_data) > 0 and np.max(np.abs(audio_data)) > 0:
                audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            wavfile.write(wav_path, 44100, audio_data)
        except Exception as e:
            print(f"Error generating missing wav: {e}")
            wav_path = None
            
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        fig = plot_piano_roll(pm, title=f"Playback: {os.path.basename(midi_path)}")
    except Exception as e:
        fig = None
        
    return wav_path, fig, midi_path

with gr.Blocks(title="MicroMusicGPT - Milestone 1 Demo") as demo:
    gr.Markdown("# MicroMusicGPT Control Panel")
    gr.Markdown("Autoregressive PyTorch Transformer with Contextual Seed Prompting.")
    
    with gr.Tabs():
        with gr.Tab("Generate New Sequence"):
            with gr.Row():
                with gr.Column():
                    seed_dropdown = gr.Dropdown(choices=get_seed_files(), value="None (Empty Start)", label="Context Prompt (Seed MIDI file)")
                    tokens_slider = gr.Slider(minimum=100, maximum=2000, value=800, step=100, label="Tokens to Generate")
                    generate_btn = gr.Button("Generate Composition", variant="primary")
                    
                with gr.Column():
                    audio_out = gr.Audio(label="Synthesized Audio output (.wav)")
                    midi_dl = gr.File(label="Generated Standard MIDI Sequence")
                    
            piano_roll_plot = gr.Plot(label="Structural MIDI Piano Roll Map")
            
        with gr.Tab("Playback Gallery"):
            gr.Markdown("Listen back to sequences previously synthesized by the model (including runs from Colab).")
            with gr.Row():
                with gr.Column():
                    gallery_dropdown = gr.Dropdown(choices=get_generated_files(), label="Select Past Generation")
                    load_btn = gr.Button("Load Sequence", variant="secondary")
                with gr.Column():
                    gallery_audio = gr.Audio(label="Synthesized Audio output (.wav)")
                    gallery_midi = gr.File(label="Generated Standard MIDI Sequence")
                    
            gallery_plot = gr.Plot(label="Structural MIDI Piano Roll Map")

    def refresh_dropdowns():
        return gr.Dropdown(choices=get_seed_files()), gr.Dropdown(choices=get_generated_files())
        
    demo.load(fn=refresh_dropdowns, inputs=[], outputs=[seed_dropdown, gallery_dropdown])
    
    generate_btn.click(fn=generate_interface, inputs=[tokens_slider, seed_dropdown], outputs=[audio_out, piano_roll_plot, midi_dl])
    load_btn.click(fn=load_generated_sequence, inputs=[gallery_dropdown], outputs=[gallery_audio, gallery_plot, gallery_midi])

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
