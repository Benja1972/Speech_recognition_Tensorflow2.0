import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import sample 

import librosa
import librosa.display
from os.path import join 
import glob
import os 


SAMPLING_RATE=16000
DUR_WAV = 22050

NUM_MFCC = 12

words = ['one', 'two', 'three', 'four', 'five']


df  = pd.DataFrame()

for wr in words:

    flph = join("../data",wr)

    wf = [f for f in glob.glob(join(flph,"*.wav"))]
    #wf = sample(wf,1000) ####### REMOVE ###############

    w_df = pd.DataFrame({'file':wf, 'labels':[wr]*len(wf)})
    
    df = pd.concat([df, w_df])
    
    df = df.sample(frac=1)


def mfccf(raw, n_mfccs=16):
    wave, sample_rate = librosa.load(raw['file'], sr = SAMPLING_RATE, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=wave, sr=SAMPLING_RATE, n_mfcc=n_mfccs)
    mfccf = np.mean(mfccs.T,axis=0)
    return mfccf

def mfccs(raw, n_mfccs=16):
    wave, sample_rate = librosa.load(raw['file'], sr = SAMPLING_RATE, res_type='kaiser_fast')
    if len(wave) != DUR_WAV:
        wave = librosa.effects.time_stretch(wave, len(wave)/DUR_WAV)
    mfccs = librosa.feature.mfcc(y=wave, sr=SAMPLING_RATE, n_mfcc=n_mfccs)

    return mfccs
    

Feature = lambda raw: mfccs(raw, n_mfccs=NUM_MFCC)

df['feats'] = df.apply(Feature, axis=1)

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


X = np.array(df['feats'].tolist())
y = np.array(df['labels'].tolist())


## Normalization (<--??? https://www.kaggle.com/c/freesound-audio-tagging/discussion/54082)
#
#~ mean = np.mean(X, axis=2, keepdims=True)
#~ std = np.std(X, axis=2, keepdims=True)
#~ X = (X - mean)/std
#~ for i in range(X.shape[2]): 
    #~ X[:,:,i] = (X[:,:,i]-mean)/std 


mean = np.mean(X)
std = np.std(X)
X = (X - mean)/std

lb = LabelEncoder()

y = tf.keras.utils.to_categorical(lb.fit_transform(y))



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if True:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten


    num_labels = y.shape[1]

    # build model
    model = Sequential()
    model.add(Flatten(input_shape=(NUM_MFCC, 44)))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels, activation="softmax"))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    history = model.fit(X_train, y_train, 
                        batch_size=100, 
                        epochs=60, 
                        validation_data=(X_test, y_test))
    
    ## ==  Verify
    # Plot training & validation accuracy values
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Validation and accuracy values')

    
    ax1.plot(history.history['accuracy'], 'o-', label='Training accuracy')
    ax1.plot(history.history['val_accuracy'], 'o-', label='Validation accuracy')
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['Train', 'Val'], loc='upper left')


    # Plot training & validation loss values
    ax2.plot(history.history['loss'], 'o-', label='Training loss')
    ax2.plot(history.history['val_loss'], 'o-', label='Validation loss')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Val'], loc='upper left')
    

    from mlxtend.plotting import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix

    y_pred = model.predict_classes(X_test)

    mat = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    plot_confusion_matrix(conf_mat=mat, 
                          #figsize=(12, 12), 
                          class_names = words, 
                          show_normed=False)

    plt.show()
