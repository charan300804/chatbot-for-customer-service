# chatbot-for-customer-service
# Intelligent Customer Support Chatbot

This project involves creating an intelligent customer support chatbot that can understand user queries and provide accurate responses. The chatbot uses NLP techniques such as text classification, named entity recognition (NER), and language models.

## Technologies Used

- **NLP**: SpaCy, Hugging Face’s Transformers (BERT, GPT-3) for language understanding and text generation.
- **Text Classification**: Logistic Regression, SVM for classifying queries.
- **Intent Recognition**: Identifying user intent and responding appropriately using NLP techniques.
- **Dialog Management**: Dialogflow for handling conversation flows.
- **Backend**: FastAPI for building an API to serve the chatbot.
- **Data**: Use datasets like the Cornell Movie Dialogs or Customer Support on Twitter dataset for training.

## Setup and Installation

1. **Clone the Repository**:
   ```sh
   git clone <repository-url>
   cd <repository-name>
Install Dependencies:

sh
pip install -r requirements.txt
Run the Chatbot:

sh
python chatbot.py
Project Structure
<repository-name>/
│
├── chatbot.py             # Main chatbot script
├── intents.json           # Intent definitions
└── README.md              # Project documentation
Usage
Start the Chatbot: Run the chatbot.py script to start the chatbot.

sh
python chatbot.py
Interact with the Chatbot:

You can interact with the chatbot through the console.

Type your queries and the chatbot will respond.

Type "quit" to exit the chatbot.

Example Interaction
You: Hello
RaMona: Hi there! How can I help you today?

You: What's the weather like today?
RaMona: I'm sorry, I don't understand. Can you please rephrase?

You: Quit
Code Explanation
Main Script (chatbot.py)
Import Necessary Libraries:

python
import numpy as np
import json
import random
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
Load Intents:

python
with open('intents.json', 'r') as file:
    intents = json.load(file)['intents']
Initialize Lemmatizer and Tokenizer:

python
lemmatizer = WordNetLemmatizer()
tokenizer = word_tokenize
Preprocess Data:

python
# Create a list of all words in the intents, and a list of all intents
words = []
classes = []
documents = []
for intent in intents:
    for pattern in intent['patterns']:
   # Tokenize and lemmatize each word in the pattern
        words_in_pattern = tokenizer(pattern.lower())
        words_in_pattern = [lemmatizer.lemmatize(word) for word in words_in_pattern]
   # Add the words to the list of all words
        words.extend(words_in_pattern)
   # Add the pattern and intent to the list of all documents
        documents.append((words_in_pattern, intent['tag']))
   # Add the intent to the list of all intents
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
Create Training Data:

python
# Remove duplicates and sort the words and classes
words = sorted(list(set(words)))
classes = sorted(classes)

# Create training data as a bag of words
training_data = []
for document in documents:
    bag = []
   # Create a bag of words for each document
    for word in words:
        bag.append(1) if word in document[0] else bag.append(0)
  # Append the bag of words and the intent tag to the training data
    output_row = [0] * len(classes)
    output_row[classes.index(document[1])] = 1
    training_data.append([bag, output_row])

# Shuffle the training data and split it into input and output lists
random.shuffle(training_data)
training_data = np.array(training_data, dtype=object)
train_x = list(training_data[:, 0])
train_y = list(training_data[:, 1])
Define and Train the Neural Network Model:

python
# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)
Define a Function to Process User Input and Generate a Response:

python
def get_response(user_input):
  # Tokenize and lemmatize the user input
    words_in_input = tokenizer(user_input.lower())
    words_in_input = [lemmatizer.lemmatize(word) for word in words_in_input]

   # Create a bag of words for the user input
    bag = [0] * len(words)
    for word in words_in_input:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1

  # Predict the intent of the user input using the trained model
    results = model.predict(np.array([bag]), verbose=0)[0]
   # Get the index of the highest probability result
    index = np.argmax(results)
  # Get the corresponding intent tag
    tag = classes[index]

   # If the probability of the predicted intent is below a certain threshold, return a default response
    if results[index] < 0.5:
        return "I'm sorry, I don't understand. Can you please rephrase?"

   # Get a random response from the intent
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])

    return response
Main Loop to Get User Input and Generate Responses:

python
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = get_response(user_input)
    print("RaMona:", response)
