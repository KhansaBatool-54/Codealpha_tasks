# music_app.py — Flask server for Music Generation UI

from flask import Flask, render_template, send_file, jsonify
import pickle
import numpy as np
from music21 import stream, note, chord
import os

app = Flask(__name__)

def generate_music():
    with open('notes.pkl', 'rb') as f:
        notes = pickle.load(f)
    with open('note_mappings.pkl', 'rb') as f:
        note_to_int, int_to_note = pickle.load(f)

    transition = np.load('transition_matrix.npy')
    n_vocab = len(int_to_note)

    current = np.random.randint(0, n_vocab)
    generated = []

    for i in range(200):
        probs = transition[current]
        if probs.sum() == 0:
            current = np.random.randint(0, n_vocab)
        else:
            temperature = 0.8
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / probs.sum()
            current = np.random.choice(n_vocab, p=probs)
        generated.append(int_to_note[current])

    output_stream = stream.Stream()
    offset = 0

    for pattern in generated:
        try:
            if '.' in pattern:
                chord_notes = [note.Note(int(n)) for n in pattern.split('.')]
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

@app.route('/')
def home():
    return render_template('music.html')

@app.route('/generate')
def generate():
    try:
        generate_music()
        return jsonify({'status': 'success', 'message': 'Music generated successfully!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download')
def download():
    if os.path.exists('generated_music.mid'):
        return send_file('generated_music.mid', as_attachment=True)
    return jsonify({'status': 'error', 'message': 'Generate music first!'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)