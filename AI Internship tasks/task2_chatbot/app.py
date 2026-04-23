from flask import Flask, request, jsonify, render_template
from chatbot import get_answer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'answer': 'Please ask a question!'})
    answer = get_answer(user_message)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
