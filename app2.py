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
from pathlib import Path
import urllib.request
import shutil

# 1. NLTK Data Setup with Robust Error Handling
def setup_nltk_data():
    """Ensure all required NLTK data is properly installed"""
    try:
        # Set up paths
        nltk_data_dir = Path.cwd() / "nltk_data"
        nltk_data_dir.mkdir(exist_ok=True)
        
        # Configure NLTK to only use our directory
        nltk.data.path = [str(nltk_data_dir)]
        
        # Download core resources
        core_resources = ['punkt', 'wordnet', 'omw-1.4']
        for resource in core_resources:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(resource, download_dir=str(nltk_data_dir))
        
        # Special handling for punkt_tab
        punkt_tab_path = nltk_data_dir / "tokenizers" / "punkt_tab"
        if not punkt_tab_path.exists():
            st.info("Downloading punkt_tab resources...")
            punkt_tab_url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
            zip_path = nltk_data_dir / "punkt_tab.zip"
            
            urllib.request.urlretrieve(punkt_tab_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(punkt_tab_path)
            os.remove(zip_path)
        
        # Verify everything is installed
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        return True
        
    except Exception as e:
        st.error(f"Failed to setup NLTK data: {str(e)}")
        return False

# 2. Initialize NLTK with error handling
if not setup_nltk_data():
    st.error("Critical error: Could not initialize NLTK data. Please check the error message above.")
    st.stop()


# Initialize NLTK with absolute path handling
def initialize_nltk():
    # Get absolute path to nltk_data directory
    nltk_data_dir = os.path.join(os.path.abspath(os.getcwd()), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download core NLTK data
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    nltk.download('omw-1.4', download_dir=nltk_data_dir)
    
    # Handle punkt_tab specifically
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        import urllib.request
        punkt_tab_url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
        urllib.request.urlretrieve(punkt_tab_url, "punkt_tab.zip")
        
        with zipfile.ZipFile("punkt_tab.zip", 'r') as zip_ref:
            zip_ref.extractall(os.path.join(nltk_data_dir, "tokenizers", "punkt_tab"))
        
        os.remove("punkt_tab.zip")
    
    # Set NLTK path
    nltk.data.path = [nltk_data_dir]
    
    # Verify
    try:
        nltk.data.find('tokenizers/punkt')
        st.success("NLTK data loaded successfully!")
    except LookupError as e:
        st.error(f"NLTK data missing: {str(e)}")

# Author Information
AUTHOR = "Streamlit Chatbot Generator by [Shailendra Singh](https://ssinghportfolio.netlify.app/)"

# Function to preprocess JSON file
def preprocess_json(file_path):
    lemmatizer = WordNetLemmatizer()
    try:
        with open(file_path, 'r') as file:
            intents = json.load(file)
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return None, None

    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.', ',']

    st.info("🔄 **Preprocessing the uploaded JSON file...**")
    progress = st.progress(0)

    try:
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
        
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return None, None

# Function to train and save the model
def train_model(train_x, train_y):
    st.info("🚀 **Training the chatbot model...**")
    with st.spinner("This may take a few moments..."):
        try:
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
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")

# Function to create ZIP file
def create_zip_file(output_filename, files):
    try:
        with zipfile.ZipFile(output_filename, 'w') as zipf:
            for file in files:
                if os.path.exists(file):
                    zipf.write(file)
        return output_filename
    except Exception as e:
        st.error(f"Error creating ZIP file: {str(e)}")
        return None

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
        
        if train_x is not None and train_y is not None:
            # Step 2: Train the model
            st.subheader("Step 2: Training the Model")
            train_model(train_x, train_y)

            # Step 3: Create ZIP file for all generated files
            st.subheader("Step 3: Download All Files")
            generated_files = ["words.pkl", "classes.pkl", "train_x.npy", "train_y.npy", "chatbot_model.keras"]
            existing_files = [f for f in generated_files if os.path.exists(f)]
            
            if len(existing_files) == len(generated_files):
                zip_filename = "chatbot_files.zip"
                if create_zip_file(zip_filename, existing_files):
                    with open(zip_filename, "rb") as zipf:
                        st.download_button(
                            label="📥 Download All Generated Files (ZIP)",
                            data=zipf,
                            file_name=zip_filename,
                            mime="application/zip"
                        )
                    st.success("✅ All files are bundled into a ZIP file and ready for download!")
            else:
                missing_files = set(generated_files) - set(existing_files)
                st.error(f"Missing generated files: {', '.join(missing_files)}")

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
        st.error(f"❌ An error occurred: {str(e)}")
else:
    st.warning("⚠️ Please upload a valid JSON file to proceed.")