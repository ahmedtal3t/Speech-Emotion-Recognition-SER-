#%%
#Libaries

import os
from IPython.display import Audio
import librosa
import librosa.display as disp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Flatten, BatchNormalization
from keras.models import Sequential
from keras.layers.convolutional import Conv1D

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import matplotlib.pyplot as plt

from keras.models import load_model

import pandas as pd

from sklearn.metrics import confusion_matrix

import seaborn as sns

import winsound
import shutil
import soundfile as sf

#%%
#Path

path = "C:\\emotional\\"

#%%
#Implementation

def load_data(path):
    f_emotions = []
    f_paths = []

    folders = os.listdir(path)

    print("folder names: ")
    print(folders)

    for folder in folders:
        files = os.listdir(path + folder)
        for file in files:
            step = file.split(".")[0]
            step = step.split("-")[2]
            f_emotions.append(int(step))
            f_paths.append(path + folder + os.sep + file)
            
    return [f_emotions, f_paths]

def get_emotion(number):
    info = {1 : "neutral",
            2 : "calm",
            3 : "happy",
            4 : "sad",
            5 : "angry",
            6 : "fearful",
            7 : "disgust",
            8 : "surprised"}
    return info[number]

emotions, pathes = load_data(path)

#%%
#Analyze and edit audio files

def read_audio(path):
    data, sample_rate = librosa.load(path,duration=2.4,offset=0.6)
    return data, sample_rate

def draw_wave(path, data, sr):
    plt.figure(figsize=(10,4))
    # data, sr = librosa.load(path)
    plt.title("Audio Wave :: "+ path,size=17)
    disp.waveshow(data,sr=sr)
    plt.show()
    
def drow_spectogram(path, data, sr):
    plt.figure(figsize=(10,4))
    # data, sr = librosa.load(path)
    X = librosa.stft(data)
    Xdp = librosa.amplitude_to_db(abs(X))
    plt.title("Spectrogram for Wave :: "+ path,size=17)
    disp.specshow(Xdp,sr=sr, x_axis = "time", y_axis = "hz")
    plt.show()

def add_noise(data, sr):
    noise = 0.035*np.random.uniform()*np.max(data)
    data += noise * np.random.normal(size=data.shape[0])
    
    return data, sr


def shift(data, sr):
    sift_range = int(np.random.uniform(low=-5, high=5)*1000)
    shifted = np.roll(data, sift_range)
    
    return shifted, sr

def pitch(data, sr, factor = 0.7):
    pitched= librosa.effects.pitch_shift(y=data, sr=sr, n_steps=factor)

    return pitched, sr

def strech(data,sr,rate=0.85):
    streched = librosa.effects.time_stretch(y=data, rate=rate)
    
    return streched, sr

#%%
#Feature Extracyion MFCCs

def feature_extracyion(data, sr):
    mfcc = librosa.feature.mfcc(y=data, sr = sr)
    
    return mfcc


def processing_audio(data, sr, option):
    
    func = random.choice(option)
    
    if func == "Standerd":
        processed = data
    else:
        processed, _ = func(data, sr)
        
    return processed



#%%


def get_features(path):
    data,sample_rate = read_audio(path)
    
    funcs=["Standerd", add_noise,pitch]       
    
    features = []
    
    func1_data = processing_audio(data, sample_rate, funcs)
    func2_data = processing_audio(func1_data, sample_rate, funcs)
    
    feature =  feature_extracyion(func2_data, sample_rate)
    if feature.shape == (20,104):
        features.append(feature)
    
    func1_data = processing_audio(data, sample_rate, funcs)
    func2_data = processing_audio(func1_data, sample_rate, funcs)
    
    feautre =  feature_extracyion(func2_data, sample_rate)
    
    if feature.shape == (20,104):
        features.append(feature)
        
    func1_data = processing_audio(data, sample_rate, funcs)
    func2_data = processing_audio(func1_data, sample_rate, funcs)
        
    feautre =  feature_extracyion(func2_data, sample_rate)
        
    if feature.shape == (20,104):
        features.append(feature)
    
    return np.array(features)

#%%

def display(number):
    path = pathes[number]
    data, sample_rate = read_audio(path)
    mfcc_features = feature_extracyion(data, sample_rate)
    print(mfcc_features)
    print("len of the mfccs= ", len(mfcc_features))
    print(get_emotion(emotions[number]))
    
    draw_wave(path,data,sample_rate)
    drow_spectogram(path,data,sample_rate)
    
    data,sample_rate = add_noise(data,sample_rate)
    data,sample_rate = shift(data,sample_rate)
    data,sample_rate = pitch(data,sample_rate)
    data,sample_rate = strech(data,sample_rate)
    mfcc_features = feature_extracyion(data, sample_rate)

    drow_spectogram(path,data,sample_rate)   
    
    source_path = pathes[number]  # المسار الخاص بالملف الصوتي
    destination_path = "C:/Users/master/OneDrive/Desktop/main_audio.wav"  # المسار الجديد للملف الصوتي
    shutil.copy(source_path, destination_path)  # نسخ الملف الصوتي
    
    # تحديد المسار الجديد للملف الصوتي
    destination_path = os.path.join(os.path.expanduser("~"), "C:/Users/master/OneDrive/Desktop/new_audio.wav")
    # حفظ الملف الصوتي المعدل باستخدام مكتبة soundfile
    sf.write(destination_path, data, sample_rate)
    
    return data, sample_rate


d, sr = display(70)
Audio(data=d, rate=sr)



#%%

X=[]
Y=[]

for indx in range(len(pathes)):
    value = get_features(pathes[indx])
    if len(value) > 0:
        for item in value:
            X.append(item)
            Y.append(np.eye(8)[emotions[indx] - 1])
        
#%%

encoder = OneHotEncoder()
encoder.fit_transform(np.array([1,2,3,4,5,6,7,8]).reshape(-1, 1)).toarray()

#%%

x_train, x_test, y_train, y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=11, shuffle=True)       
print("train x shape ", x_train.shape)
print("test y shape ", x_test.shape)
print("train x shape ", y_train.shape)
print("test y shape ", y_train.shape)

#%%


trainX = np.expand_dims(x_train, axis=3)
trainX = np.expand_dims(x_train, axis=3)
trainX = np.swapaxes(trainX, 1, 2)
print("trainX: ", trainX.shape)


testX = np.expand_dims(x_test, axis=3)
testX = np.expand_dims(testX, axis=3)
testX = np.swapaxes(testX, 1, 2)
print("testX: ", testX.shape)


inputShape = trainX.shape[1:]

print(inputShape)


#%%

def createModel(inputShape):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(32, kernel_size=3, padding="same", activation="relu"), input_shape=inputShape))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Flatten()))
    
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=64, activation="relu"))
    
    model.add(Dense(units=8, activation="softmax"))

    return model


model = createModel(inputShape)

model.summary()

#%%
#Training Tje Model

opt = Adam(learning_rate=0.01)

model.compile(optimizer=opt, loss="categorical_crossentropy",
              metrics=["accuracy"])

reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.6, verbose=1, patience=5, min_lr=1e-8)
stop = EarlyStopping(monitor="val_loss", patience=7)

hist = model.fit(trainX, y_train, batch_size=140, epochs=80, validation_data=(testX, y_test), callbacks=[reduce, stop])


#%%
#Testing The Model

#print("The accuracy",model.evaluate(testX, y_test)[1]*100,"%")

train_loss = hist.history["loss"]
test_loss = hist.history["val_loss"]
train_accuracy = hist.history["accuracy"]
test_accuracy = hist.history["val_accuracy"]

epochs = [value for value in range(80)]


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(15, 6)
ax[0].plot(epochs, train_loss, label="Traning Loss")
ax[0].plot(epochs, test_loss, label="Testing Loss")
ax[0].set_title(" Testing & Traning Loss")
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(epochs, train_accuracy, label="Traning Accuracy")
ax[1].plot(epochs, test_accuracy, label="Testing Accuracy")
ax[1].set_title(" Testing & Traning Accuracy")
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")

#%%

model.save("C:\\emotional\\emotion_model.h5")

#%%
#Testing

emotion_model = load_model("C:\\emotional\\emotion_model.h5")

y_pred = emotion_model.predict(testX)

#%%

predY = encoder.inverse_transform(y_pred)
testY = encoder.inverse_transform(y_test)

emotions_predict = [get_emotion(value) for value in predY.flatten()]
emotions_actual = [get_emotion(value) for value in testY.flatten()]

df = pd.DataFrame(columns=["Actual Emotion", "Predicted Emotion"])

df["Actual Emotion"] = emotions_actual
df["Predicted Emotion"] = emotions_predict

df.head(10)

#%%
#Confusion Matrix
cm = confusion_matrix(testY, predY)

emts = [get_emotion(em) for em in encoder.categories_[0]]

cmt = pd.DataFrame(cm, index=emts, columns=emts)
plt.figure(figsize=(12,10))

sns.heatmap(cmt, annot=True, fmt="", cmap="Blues")
plt.title("Confution Matrix", size=15)
plt.xlabel("Predicted Matrix", size=15)
plt.ylabel("Actual Matrix", size=15)

