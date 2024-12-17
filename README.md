# 🤖 Streamlit Chatbot Builder  

**Streamlit Chatbot Builder** is an intuitive tool that allows you to quickly create and train chatbots using a JSON intents file. This app handles everything: preprocessing, model training, and file generation. Use the sample code provided to integrate and deploy your chatbot locally.

---

## 📋 Table of Contents  

1. [Features](#features)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [How to Use](#how-to-use)  
5. [Intents JSON Format](#intents-json-format)  
6. [Sample Chatbot Code](#sample-chatbot-code)  
7. [Files Generated](#files-generated)  
8. [Project Demo](#project-demo)  
9. [Contributing](#contributing)  
10. [License](#license)  
11. [Author](#author)  

---

## 🚀 Features  

- 📤 **Upload Intents File**: Upload a valid chatbot intents file in JSON format.  
- 🔄 **Automatic Preprocessing**: Tokenizes, lemmatizes, and processes your data.  
- 🧠 **Model Training**: Trains a deep learning model (Keras Sequential).  
- 💾 **Downloadable Files**: Generates and provides the following files in a ZIP archive:  
   - `words.pkl`  
   - `classes.pkl`  
   - `train_x.npy`  
   - `train_y.npy`  
   - `chatbot_model.keras`  
- 📝 **Sample Code**: Provides ready-to-use code to deploy your chatbot.  
- 🎈 **Streamlit UI**: Interactive and user-friendly interface with progress updates.  

---

## 🛠️ Requirements  

To run this project, you need:

- **Python 3.8+**  
- Libraries listed in the `requirements.txt`.  

### **Install Dependencies**  

Run the following command to install required libraries:  

```bash
pip install -r requirements.txt
💻 Installation
Follow these steps to set up the project:

Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/streamlit-chatbot-builder.git
cd streamlit-chatbot-builder
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Run the Streamlit Application

bash
Copy code
streamlit run chatbot_builder.py
Open in Browser
The app will open in your browser, or you can visit:

arduino
Copy code
http://localhost:8501
📤 How to Use
Upload a JSON Intents File

Click on the "Upload" button to upload your JSON file containing intents.
Automatic Processing

The app tokenizes, lemmatizes, and processes your data. Progress will be displayed in real-time.
Train the Model

The app trains a Keras-based deep learning model using the processed data.
Download Files

A ZIP archive containing all generated files will be available for download.
Use the Sample Code

Copy the provided sample code to interact with your trained chatbot locally.
📝 Intents JSON Format
Your intents JSON file should follow this format:

json
Copy code
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello!", "Hi there!", "Greetings!"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you", "Goodbye"],
      "responses": ["Goodbye!", "See you soon!", "Take care!"]
    }
  ]
}
Each intent contains:

tag: A unique identifier for the intent.
patterns: User input examples.
responses: Bot responses.
