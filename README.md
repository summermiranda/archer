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
from elevenlabs import play, save, stream, Voice, VoiceSettings

# Set your Eleven Labs API key and OpenWeatherMap API key
client = ElevenLabs(api_key='839b7392885bf1a0d1914420cc55fde9')
weather_api_key = 'd6a10917ee5dd0d3affb382ba5ebf901'  # Replace with your API key

# Initialize the Wav2Vec2Processor with the 'facebook/wav2vec2-base-960h' model
tokenizer = transformers.Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

# Preload common audio files during initialization
hello_text = 'Hello! How can I assist you?'
joke_text = "Why don't scientists trust atoms? Because they make up everything."

# Load the pre-trained model and tokenizer for speech recognition
speech_recognition_model = "facebook/wav2vec2-base-960h"
tokenizer = transformers.Wav2Vec2CTCTokenizer.from_pretrained(speech_recognition_model)
model = transformers.Wav2Vec2ForCTC.from_pretrained(speech_recognition_model)

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key'

# Define additional ML categories and train a neural network classifier
vectorizer = CountVectorizer()
X_train_extended = vectorizer.fit_transform(
    ["hello", "tell me a joke", "goodbye", "weather", "play music", "set reminder", "send email"])
y_train_extended = ["hello", "joke", "goodbye", "weather", "music", "reminder", "email"]
ml_model_extended = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
ml_model_extended.fit(X_train_extended, y_train_extended)

# Define RL parameters
num_actions = len(y_train_extended)
num_states = 2  # Define states based on context, e.g., previous user interactions
num_episodes = 1000  # Number of episodes for training
max_steps_per_episode = 100  # Maximum number of steps per episode
exploration_rate = 1  # Initial exploration rate
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
learning_rate = 0.1  # Learning rate
discount_factor = 0.99  # Discount factor for future rewards

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
            # Select action using epsilon-greedy policy
            action = select_action(state)

            # Execute action and observe reward and next state
            # For demonstration purposes, assume reward and next state are predefined
            reward = np.random.random()  # Placeholder for actual reward
            next_state = np.random.choice(num_states)  # Placeholder for actual next state
            done = step == max_steps_per_episode - 1  # Placeholder for actual done condition

# Update Q-table based on Q-learning update rule
            update_Q_table(state, action, reward, next_state)

            # Update total rewards
            total_rewards += reward

            # Update state for next step
            state = next_state

            # Check if episode is done
            if done:
                break

        # Update exploration rate
        exploration_rate = min_exploration_rate + \
                           (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    return total_rewards

# Function to interact with the user and update RL
def process_command_rl(command, state):
    if not assistant_speaking:
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

        # Interact with the user using RL to update the assistant's behavior
        total_rewards = interact_with_user_rl(state)
        # Update state based on context, e.g., previous user interactions

        # Log total rewards or other metrics for analysis

        # Update RL hyperparameters if needed, e.g., learning rate, discount factor

        # Update exploration rate

        # Return total rewards or other relevant information for monitoring or analysis


# Add a flag to track whether the assistant is currently speaking
assistant_speaking = False


# Function to generate and play audio using Eleven Labs
def generate_and_play(text, voice='Thomas', model='eleven_monolingual_v1'):
    global assistant_speaking
    assistant_speaking = True  # Set the flag when the assistant starts speaking
    audio = elevenlabs.generate(text=text, voice=voice, model=model)
    elevenlabs.play(audio)
    assistant_speaking = False  # Clear the flag when the assistant finishes speaking


# Define 'city' as a global variable
city = None


# Function to process user command
def process_command(command):
    global city

    if not assistant_speaking:
        if 'hello' in command and 'how can i assist you' not in command:
            generate_and_play(hello_text)
        elif 'tell me a joke' in command:
            generate_and_play(joke_text)
        elif 'goodbye' in command:
            generate_and_play('Goodbye! Have a nice day.')
            exit()
        elif 'weather' in command:
            city = command.split('weather in ')[-1]  # Define 'city' based on user input
            get_weather(city)
        elif 'play music' in command:
            play_music()
        else:
            ml_class = ml_model_extended.predict(vectorizer.transform([command]))[0]
            if ml_class == "weather":
                city = command.split('weather in ')[-1]  # Define 'city' based on user input
                get_weather(city)
            elif ml_class == "music":
                play_music()
            else:
                generate_and_play("I'm sorry, I didn't understand that command.")
            # Interact with the user using RL to update the assistant's behavior
            process_command_rl(command, state)


# Function to get weather updates
def get_weather(city):
    base_url = (f"http://api.openweathermap.org/data/2.5/weather?q={city}"
                f"&appid={weather_api_key}&units=metric")

    response = requests.get(base_url)
    data = response.json()

    if response.status_code == 200:
        temperature = data['main']['temp']
        description = data['weather'][0]['description']

        # Generate weather text using 'city' variable
        weather_text = (f"The current temperature in {city} is {temperature} degrees Celsius with {description}.")
        generate_and_play(weather_text)

    else:
        print(f"API request failed with status code {response.status_code}")
        print("API response:")
        print(response.text)
        generate_and_play("Sorry, I couldn't fetch the weather information.")

# Function to play music
def play_music():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load('/Users/deformedbouquet/Downloads/fineline.mp3')
        pygame.mixer.music.play()

        # Let the music play for a while
        pygame.time.wait(9000)  # 9000 milliseconds (9 seconds)

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

            # Process the command using threading
            threading.Thread(target=process_command, args=(command,)).start()
    except sr.UnknownValueError:
        print("I didn't quite catch that. Can you say it again?")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f'An error occurred: {e}')
