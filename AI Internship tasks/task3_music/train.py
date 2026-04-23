# train.py — NumPy only (no PyTorch needed!)

import pickle
import numpy as np

print("Loading notes...")
with open('notes.pkl', 'rb') as f:
    notes = pickle.load(f)

unique_notes = sorted(set(notes))
n_vocab = len(unique_notes)
print(f"Unique notes: {n_vocab}")

note_to_int = {note: num for num, note in enumerate(unique_notes)}
int_to_note = {num: note for num, note in enumerate(unique_notes)}

with open('note_mappings.pkl', 'wb') as f:
    pickle.dump((note_to_int, int_to_note), f)

print("\nLearning note patterns...")

# Transition matrix — konsa note ke baad konsa note aata hai
transition = np.zeros((n_vocab, n_vocab), dtype=np.float32)

for i in range(len(notes) - 1):
    current = note_to_int[notes[i]]
    next_n  = note_to_int[notes[i + 1]]
    transition[current][next_n] += 1

# Normalize
row_sums = transition.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
transition = transition / row_sums

np.save('transition_matrix.npy', transition)
print("Model saved!")
print("Now run: py -3.13 generate.py")