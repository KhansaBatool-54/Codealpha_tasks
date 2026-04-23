# generate.py — Naya music generate karo (no PyTorch needed!)

import pickle
import numpy as np
from music21 import stream, note, chord

print("Loading model...")
with open('notes.pkl', 'rb') as f:
    notes = pickle.load(f)

with open('note_mappings.pkl', 'rb') as f:
    note_to_int, int_to_note = pickle.load(f)

transition = np.load('transition_matrix.npy')
n_vocab = len(int_to_note)

print("Generating music...")

# Random start note
current = np.random.randint(0, n_vocab)
generated = []
generate_length = 200

for i in range(generate_length):
    probs = transition[current]
    
    if probs.sum() == 0:
        current = np.random.randint(0, n_vocab)
    else:
        # Temperature sampling
        temperature = 0.8
        probs = np.power(probs, 1.0 / temperature)
        probs = probs / probs.sum()
        current = np.random.choice(n_vocab, p=probs)
    
    generated.append(int_to_note[current])
    
    if (i+1) % 50 == 0:
        print(f"Generated {i+1}/{generate_length} notes...")

# Convert to MIDI
print("\nConverting to MIDI...")
output_stream = stream.Stream()
offset = 0

for pattern in generated:
    try:
        if '.' in pattern:
            notes_in_chord = pattern.split('.')
            chord_notes = []
            for n in notes_in_chord:
                new_note = note.Note(int(n))
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_stream.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_stream.append(new_note)
        offset += 0.5
    except:
        continue

output_stream.write('midi', fp='generated_music.mid')
print("\nMusic saved as: generated_music.mid")
print("Open with VLC Player to listen! 🎵")