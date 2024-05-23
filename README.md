import speech_recognition as sr
import threading
import requests
import pygame
import transformers
import torch
import openai
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from elevenlabs.client import ElevenLabs

# Set your API keys
weather_api_key = 'your_weather_api_key'
openai.api_key = 'your_openai_api_key'
client = ElevenLabs(api_key='your_elevenlabs_api_key')

# Initialize the Wav2Vec2Processor with the 'facebook/wav2vec2-base-960h' model
tokenizer = transformers.Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

# Preload common audio files during initialization
hello_text = 'Hello! How can I assist you?'
joke_text = "Why don't scientists trust atoms? Because they make up everything."

# Load the pre-trained model and tokenizer for speech recognition
speech_recognition_model = "facebook/wav2vec2-base-960h"
tokenizer = transformers.Wav2Vec2CTCTokenizer.from_pretrained(speech_recognition_model)
model = transformers.Wav2Vec2ForCTC.from_pretrained(speech_recognition_model)

# Define additional ML categories and train a neural network classifier
vectorizer = CountVectorizer()
X_train_extended = vectorizer.fit_transform(
    ["hello", "tell me a joke", "goodbye", "weather", "play music", "set reminder", "send email"])
y_train_extended = ["hello", "joke", "goodbye", "weather", "music", "reminder", "email"]
ml_model_extended = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
ml_model_extended.fit(X_train_extended, y_train_extended)

# Define RL parameters
num_actions = len(y_train_extended)
num_states = 2
num_episodes = 1000
max_steps_per_episode = 100
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
learning_rate = 0.1
discount_factor = 0.99

# Initialize Q-table with zeros
Q_table = np.zeros((num_states, num_actions))

# Function to transcribe speech to text using the transformer-based model
def transcribe_speech(audio_data):
    inputs = tokenizer(audio_data, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    return transcription[0]

# Function to generate responses using GPT-3
def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=50
    )
    return response.choices[0].text.strip()

# Function to select an action using epsilon-greedy policy
def select_action(state):
    exploration_rate_threshold = np.random.uniform(0, 1)
    if exploration_rate_threshold > exploration_rate:
        return np.argmax(Q_table[state])
    else:
        return np.random.choice(num_actions)

# Function to update Q-table based on Q-learning update rule
def update_Q_table(state, action, reward, next_state):
    best_next_action = np.argmax(Q_table[next_state])
    td_target = reward + discount_factor * Q_table[next_state][best_next_action]
    td_error = td_target - Q_table[state][action]
    Q_table[state][action] += learning_rate * td_error

# Function to interact with the user using RL
def interact_with_user_rl(state):
    total_rewards = 0
    for episode in range(num_episodes):
        for step in range(max_steps_per_episode):
            action = select_action(state)
            reward = np.random.random()
            next_state = np.random.choice(num_states)
            done = step == max_steps_per_episode - 1

            update_Q_table(state, action, reward, next_state)
            total_rewards += reward
            state = next_state

            if done:
                break

        exploration_rate = min_exploration_rate + \
                           (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    return total_rewards

# Function to generate and play audio using Eleven Labs
def generate_and_play(text):
    audio = client.generate(text=text)
    client.play(audio)

# Function to process user command
def process_command(command):
    if 'hello' in command and 'how can i assist you' not in command:
        generate_and_play(hello_text)
    elif 'tell me a joke' in command:
        generate_and_play(joke_text)
    elif 'goodbye' in command:
        generate_and_play('Goodbye! Have a nice day.')
        exit()
    elif 'weather' in command:
        city = command.split('weather in ')[-1]
        get_weather(city)
    elif 'play music' in command:
        play_music()
    else:
        ml_class = ml_model_extended.predict(vectorizer.transform([command]))[0]
        if ml_class == "weather":
            city = command.split('weather in ')[-1]
            get_weather(city)
        elif ml_class == "music":
            play_music()
        else:
            generate_and_play("I'm sorry, I didn't understand that command.")

# Function to process user command and update RL
def process_command_rl(command, state):
    process_command(command)
    total_rewards = interact_with_user_rl(state)
    return total_rewards

# Function to get weather updates
def get_weather(city):
    base_url = (f"http://api.openweathermap.org/data/2.5/weather?q={city}"
                f"&appid={weather_api_key}&units=metric")

    response = requests.get(base_url)
    data = response.json()

    if response.status_code == 200:
        temperature = data['main']['temp']
        description = data['weather'][0]['description']
        weather_text = (f"The current temperature in {city} is {temperature} degrees Celsius with {description}.")
        generate_and_play(weather_text)
    else:
        generate_and_play("Sorry, I couldn't fetch the weather information.")

# Function to play music
def play_music():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load('/path/to/your/music.mp3')
        pygame.mixer.music.play()
        pygame.time.wait(9000)  # Let the music play for 9 seconds
        pygame.mixer.music.stop()
    except Exception as e:
        print(f"An error occurred while playing music: {e}")

# Main loop for speech recognition and command processing
while True:
    try:
        with sr.Microphone() as source:
            print('Listening...')
            recognizer = sr.Recognizer()
            recognizer.adjust_for_ambient_noise(source)

            audio_data = recognizer.listen(source, timeout=10)
            transcribed_text = transcribe_speech(audio_data)
            command = transcribed_text.lower()
            print(f"You said: {command}")

            threading.Thread(target=process_command_rl, args=(command, 0)).start()
    except sr.UnknownValueError:
        print("I didn't quite catch that. Can you say it again?")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f'An error occurred: {e}')

