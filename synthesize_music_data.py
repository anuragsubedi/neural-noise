import random 

# --- REPLACEMENT START ---
# Context: You are interested in Music Transformers. 
# We simulate a "Music Dataset" where characters = notes.
# c, d, e, f, g, a, b (lower octave) | C, D, E, F, G, A, B (upper octave)

print("Synthesizing musical dataset...")
docs = []
# Create 50 "songs" that are just ascending C-Major scales
for _ in range(50):
    docs.append("cdefgabC") 

# Create 50 "songs" that are descending C-Major scales
for _ in range(50):
    docs.append("CBAgfedc")

# Create 50 "songs" that are simple arpeggios (C-E-G-C)
for _ in range(50):
    docs.append("cegC")

random.shuffle(docs)
print(f"num docs: {len(docs)}")
# --- REPLACEMENT END ---

with open("input.txt", "w") as f:
    f.write("\n".join(docs))