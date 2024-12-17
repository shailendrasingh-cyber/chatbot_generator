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

- **Python 3.11.5 +**  
- Libraries listed in the `requirements.txt`.  

### **Install Dependencies**  

Make sure the json file in :-
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

##tag: A unique identifier for the intent.
##patterns: User input examples.
##responses: Bot responses.


#sample ![sample1](https://github.com/user-attachments/assets/90571b1a-3c9a-4b69-8c1a-e95c8ec099ef)


