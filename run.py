import os
import shutil
import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf


label2int = {
    "male": 1,
    "female": 0
}

def create_model(vector_length=128):
    """5 hidden dense layers from 256 units to 64, not the best model, but not bad."""
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function, 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    # print summary of the model
    model.summary()
    return model

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result


if __name__ == "__main__":
    # Configure TensorFlow to use GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # construct the model
    model = create_model()
    # load the saved/trained weights
    model.load_weights("model/model.h5")

    input_folder = './input'  # Specify the path to your input folder
    output_folder = './output'  # Specify the path to your output folder

    male_folder = os.path.join(output_folder, 'male')
    female_folder = os.path.join(output_folder, 'female')
    os.makedirs(male_folder, exist_ok=True)
    os.makedirs(female_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.ogg') or file_name.endswith('.wav'):

            file_path = os.path.join(input_folder, file_name)

            # extract features and reshape it
            features = extract_feature(file_path, mel=True).reshape(1, -1)
            # predict the gender!
            male_prob = model.predict(features)[0][0]
            female_prob = 1 - male_prob
            gender = "male" if male_prob > female_prob else "female"

            print("Result:", gender)
            print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")
            if gender == 'male':
                output_path = os.path.join(male_folder, file_name)
            else:
                output_path = os.path.join(female_folder, file_name)

            shutil.copyfile(file_path, output_path)
