# preprocess.py — MIDI files se notes extract karo

from music21 import converter, instrument, note, chord
import os
import pickle

def get_notes_from_midi(midi_folder='midi_data'):
    notes = []
    
    files = [f for f in os.listdir(midi_folder) if f.endswith('.mid') or f.endswith('.midi')]
    print(f"Found {len(files)} MIDI files!")
    
    for file in files:
        path = os.path.join(midi_folder, file)
        print(f"Processing: {file}")
        
        try:
            midi = converter.parse(path)
            parts = instrument.partitionByInstrument(midi)
            
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Skipping {file}: {e}")
    
    print(f"\nTotal notes extracted: {len(notes)}")
    
    # Save notes
    with open('notes.pkl', 'wb') as f:
        pickle.dump(notes, f)
    
    print("Notes saved to notes.pkl!")
    return notes

if __name__ == '__main__':
    get_notes_from_midi()