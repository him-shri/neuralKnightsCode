import os
from typing import Tuple
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model

from models.common import ConfidenceScore
from .config import TEST_DATA_PATH
from .config import TRAINING_FILES_PATH
from .config import MODEL_DATA_PATH

# Define your folder structure
data_dir = TRAINING_FILES_PATH
classes = ['human', 'robot']

# Load and preprocess audio data
def load_and_preprocess_data(data_dir, classes, target_shape=(128, 128)):
    data = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                # file_path = filename
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                data.append(mel_spectrogram)
                labels.append(i)
            else:
                print('not found')
    
    return np.array(data), np.array(labels)

# Split data into training and testing sets
data, labels = load_and_preprocess_data(data_dir, classes)
labels = to_categorical(labels, num_classes=len(classes))  # Convert labels to one-hot encoding
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create a neural network model
input_shape = X_train[0].shape
input_layer = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(len(classes), activation='softmax')(x)
model = Model(input_layer, output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

# Save the model
model.save(MODEL_DATA_PATH + 'audio_classification_model.h5')

# Load the saved model
model = load_model(MODEL_DATA_PATH + 'audio_classification_model.h5')

# Define the target shape for input spectrograms
target_shape = (128, 128)

# Define your class labels
classes = ['human', 'robot']

# Function to preprocess and classify an audio file
def test_audio(file_path, model):
    # Load and preprocess the audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))
    
    # Make predictions
    predictions = model.predict(mel_spectrogram)
    
    # Get the class probabilities
    class_probabilities = predictions[0]
    
    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)
    
    return class_probabilities, predicted_class_index

# Test an audio file
def testSystemProbability(audio_file_path: str)-> Tuple[ConfidenceScore, str]:
    test_audio_file = TEST_DATA_PATH + audio_file_path
    class_probabilities, predicted_class_index = test_audio(test_audio_file, model)

    # Display results for all classes
    result = ConfidenceScore(aiProbability=0, humanProbability=0, accuracy=0, spanishProbability=0, frenchProbability=0, englishProbability=0, language="")
    humanOrRobot = 'human'
    humanOrRobotPercent = -1

    for i, class_label in enumerate(classes):
        print('class->',class_label)
        if class_label == 'human':
            if class_probabilities[i] >= humanOrRobotPercent:
                humanOrRobot = 'human'
                humanOrRobotPercent = class_probabilities[i]
            result.humanProbability= class_probabilities[i] * 100
        elif class_label == 'robot':
            if class_probabilities[i] >= humanOrRobotPercent:
                humanOrRobot = 'robot'
                humanOrRobotPercent = class_probabilities[i]
            result.aiProbability = class_probabilities[i] * 100

    return result, humanOrRobot


# test_audio_file = TEST_DATA_PATH + 'TestData.wav'
# class_probabilities, predicted_class_index = test_audio(test_audio_file, model)

# # Display results for all classes
# for i, class_label in enumerate(classes):
#     probability = class_probabilities[i]
#     print(f'Class: {class_label}, Probability: {probability:.4f}')

# # Calculate and display the predicted class and accuracy
# predicted_class = classes[predicted_class_index]
# accuracy = class_probabilities[predicted_class_index]
# print(f'The audio is classified as: {predicted_class}')
# print(f'Accuracy: {accuracy:.4f}')

