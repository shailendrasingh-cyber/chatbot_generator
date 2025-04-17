from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import random
import json
from tensorflow.keras.models import load_model
import os

# Initialize app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load NLTK data path
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()

# Load model and related files
model = load_model('chatbot_model.keras')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents
with open('uploaded_intents.json') as f:
    intents = json.load(f)

# Preprocessing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I couldn't find a suitable response."

# Serve frontend if any
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Chat API
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({"error": "No message provided"}), 400

        intents_pred = predict_class(message)
        response = get_response(intents_pred, intents)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == '__main__':
    app.run(debug=True)
