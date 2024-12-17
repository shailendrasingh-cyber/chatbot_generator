import streamlit as st
import json
import os
import zipfile
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize NLTK downloads
nltk.download('punkt')
nltk.download('wordnet')

# Author Information
AUTHOR = "Streamlit Chatbot Generator by [Shailendra Singh](https://ssinghportfolio.netlify.app/)"

# Function to preprocess JSON file
def preprocess_json(file_path):
    lemmatizer = WordNetLemmatizer()
    with open(file_path, 'r') as file:
        intents = json.load(file)

    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.', ',']

    st.info("🔄 **Preprocessing the uploaded JSON file...**")
    progress = st.progress(0)

    for i, intent in enumerate(intents['intents']):
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
        progress.progress((i + 1) / len(intents['intents']))

    words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
    classes = sorted(set(classes))

    # Save words and classes to pickle files
    with open("words.pkl", "wb") as f:
        pickle.dump(words, f)
    with open("classes.pkl", "wb") as f:
        pickle.dump(classes, f)

    # Training data preparation
    training = []
    output_empty = [0] * len(classes)
    for document in documents:
        bag = []
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
        for word in words:
            bag.append(1 if word in word_patterns else 0)
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    np.save("train_x.npy", train_x)
    np.save("train_y.npy", train_y)

    st.success("✅ Preprocessing Complete! Files generated: `words.pkl`, `classes.pkl`, `train_x.npy`, `train_y.npy`.")
    return train_x, train_y

# Function to train and save the model
def train_model(train_x, train_y):
    st.info("🚀 **Training the chatbot model...**")
    with st.spinner("This may take a few moments..."):
        model = Sequential([
            Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(train_y[0]), activation='softmax')
        ])

        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=0)
        model.save("chatbot_model.keras")

    st.success("✅ Model trained and saved as `chatbot_model.keras`!")
    st.balloons()

# Function to create ZIP file
def create_zip_file(output_filename, files):
    with zipfile.ZipFile(output_filename, 'w') as zipf:
        for file in files:
            zipf.write(file)
    return output_filename

# Streamlit app layout
st.set_page_config(page_title="Chatbot Generator", layout="centered", page_icon="🤖")
st.title("🤖 Chatbot Generator")
st.markdown(AUTHOR)

# Sidebar for instructions
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.write("""
    1. Upload a **valid JSON intents file**.
    2. Wait for preprocessing to complete.
    3. The chatbot model will be trained and saved automatically.
    4. Generated files:
       - `words.pkl`
       - `classes.pkl`
       - `train_x.npy`, `train_y.npy`
       - `chatbot_model.keras`
    5. Download all files as a ZIP archive.
    """)
    st.write("✅ Enjoy building your chatbot!")

# File uploader for JSON intents file
uploaded_file = st.file_uploader("📤 **Upload your JSON Intents File**", type=['json'], key="uploader")

if uploaded_file:
    try:
        # Save uploaded file temporarily
        file_path = "uploaded_intents.json"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.success("📥 File uploaded successfully!")
        st.divider()

        # Step 1: Preprocess JSON file
        st.subheader("Step 1: Preprocessing Data")
        train_x, train_y = preprocess_json(file_path)

        # Step 2: Train the model
        st.subheader("Step 2: Training the Model")
        train_model(train_x, train_y)

        # Step 3: Create ZIP file for all generated files
        st.subheader("Step 3: Download All Files")
        generated_files = ["words.pkl", "classes.pkl", "train_x.npy", "train_y.npy", "chatbot_model.keras"]
        zip_filename = "chatbot_files.zip"
        create_zip_file(zip_filename, generated_files)

        # Provide download button for the ZIP file
        with open(zip_filename, "rb") as zipf:
            st.download_button(
                label="📥 Download All Generated Files (ZIP)",
                data=zipf,
                file_name=zip_filename,
                mime="application/zip"
            )

        st.success("✅ All files are bundled into a ZIP file and ready for download!")

        # Step 4: Display Sample Code
        st.subheader("🎉 Sample Code to Build Your Chatbot")
        st.code("""
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.keras")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

with open("uploaded_intents.json", "r") as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_data = bow(sentence, words)
    res = model.predict(np.array([bow_data]))[0]
    return classes[np.argmax(res)]

def get_response(intent_name):
    for intent in intents["intents"]:
        if intent["tag"] == intent_name:
            return random.choice(intent["responses"])

while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    intent_name = predict_class(message)
    response = get_response(intent_name)
    print(f"Bot: {response}")
        """, language="python")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.warning("⚠️ Please upload a valid JSON file to proceed.")
