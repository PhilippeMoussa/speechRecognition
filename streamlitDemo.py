# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import pickle
import matplotlib
from matplotlib import pyplot as plt
import librosa
from librosa import stft
import numpy as np
import time
import random
import tensorflow as tf
from tensorflow import keras
import os
from io import BytesIO
import streamlit.components.v1 as components


### utilitaires pour demo


def random3():
    return [int(2500*random.random()), 
              int(2500*random.random()), 
              int(2500*random.random())]

def getTranscripts(df_results, ranks, epoch1, epoch2):
    
    header1 = 'v12big_' + epoch1
    header1_d = header1 + '_d'
    header2 = 'v12big_' + epoch2
    header2_d = header2 + '_d'

    
    df = pd.DataFrame()
    
    label1 = df_results['label'].values[ranks[0]]
    score1_1 = str(round(df_results[header1_d].values[ranks[0]],2))
    pred1_1 = df_results[header1].values[ranks[0]]
    score1_2 = str(round(df_results[header2_d].values[ranks[0]],2))
    pred1_2 = df_results[header2].values[ranks[0]]
            
    label2 = df_results['label'].values[ranks[1]]
    score2_1 = str(round(df_results[header1_d].values[ranks[1]],2))
    pred2_1 = df_results[header1].values[ranks[1]]
    score2_2 = str(round(df_results[header2_d].values[ranks[1]],2))
    pred2_2 = df_results[header2].values[ranks[1]]

    label3 = df_results['label'].values[ranks[2]]
    score3_1 = str(round(df_results[header1_d].values[ranks[2]],2))
    pred3_1 = df_results[header1].values[ranks[2]]
    score3_2 = str(round(df_results[header2_d].values[ranks[2]],2))
    pred3_2 = df_results[header2].values[ranks[2]]
    
    
    df['Sample'] = ['Audio1', 'Audio2', 'Audio3']
    df['Labels'] = [label1, label2, label3]
    df['Predictions epoch ' + epoch1] = [pred1_1, pred2_1, pred3_1]
    df['Acc. ' +epoch1]= [score1_1, score2_1, score3_1]
    df['Predictions epoch '+ epoch2] = [pred1_2, pred2_2, pred3_2]
    df['Acc. ' +epoch2]= [score1_2, score2_2, score3_2]
    df.set_index('Sample')    
    return df
    
def randomDisplay(df_results, epoch1, epoch2):
     
     rank1, rank2, rank3 = random3()
     
     col1, col2, col3 = st.columns(3)
     
     with col1:
        st.write('Audio1')
        st.audio(df_results['localPath'].values[rank1])
     with col2:
        st.write('Audio2')
        st.audio(df_results['localPath'].values[rank2])
     with col3:
        st.write('Audio3')
        st.audio(df_results['localPath'].values[rank3])
     
     st.table(getTranscripts(df_results, [rank1, rank2, rank3], epoch1, epoch2))


def subsample(audioSignal, factor):
    outputSignal = [audioSignal[factor*k] for k in np.arange(0, int(len(audioSignal)/factor))]
    average = np.mean(outputSignal)
    std_deviation = np.std(outputSignal)
    outputSignal = (outputSignal-average)/std_deviation
    return outputSignal

def normalizeSignalLength(audioSignal, duration, freq):
    nbSamples = int(freq * duration)
    if len(audioSignal) > nbSamples:
        audioSignal = audioSignal[:nbSamples]
    if len(audioSignal) < nbSamples:
        audioSignal = np.concatenate((audioSignal, np.zeros(int(nbSamples -len(audioSignal)))))
    return audioSignal

def signalSpectrogram(audioSignal, freq = 16000, dt=0.025, k_temp = 1, k_freq = 1):

  spectro = np.abs(stft(audioSignal, n_fft = int(freq * dt/k_freq), 
                       hop_length = int(freq * dt * k_temp)))

  return spectro
  
def signalLogMelSpectrogram(audioSignal, freq = 16000, dt = 0.025, k_temp = 1, k_freq = 1):
   
    spectrogram = signalSpectrogram(audioSignal, dt = dt, freq = freq, k_temp = k_temp, k_freq = k_freq)
    num_spectrograms_bins = spectrogram.T.shape[-1] #soit n_fft//2 +1 = (samplingFrequency*dt/k_freq)//2 + 1
    
    linear_to_mel_weight_matrix = librosa.filters.mel(
        sr = freq,
        n_fft=int(dt*freq/k_freq) + 1,
        n_mels=num_spectrograms_bins).T

    mel_spectrogram = np.tensordot(
        spectrogram.T,
        linear_to_mel_weight_matrix,
        1)

    return np.log(mel_spectrogram + 1e-6)

def signalPredict(model, audioSignal, freq, duration):
    signal = subsample(audioSignal, factor = 3)
    normalizedAudioSignal = normalizeSignalLength(signal, duration = duration, freq = freq)
    logMel = signalLogMelSpectrogram(normalizedAudioSignal, freq = freq, k_temp = .7 , k_freq = 1.5)
    logMel = np.array([(logMel)])
    st.write(decode_batch_predictions(model.predict(logMel))[0])
     
### fin des utilitaires pour demo

class audioFile:

    def __init__(self, filename, normalize=False, root_path="", sr = 16000):

        self.audioSignal, self.samplingFrequency = librosa.load(
            path=filename, sr=sr)

        if normalize:
            average = np.mean(self.audioSignal)
            std_deviation = np.std(self.audioSignal)
            self.audioSignal = (self.audioSignal - average)/std_deviation

        self.length = len(self.audioSignal)

    def spectrogram(self, dt=0.025, k_temp=1, k_freq=1):

        spectrogram = np.abs(stft(self.audioSignal, n_fft=int(self.samplingFrequency * dt/k_freq),
                                  hop_length=int(self.samplingFrequency * dt * k_temp)))

        return spectrogram

    def logMelSpectrogram(self, dt=0.025, k_temp=1, k_freq=1):

        spectrogram = self.spectrogram(dt, k_temp, k_freq)
        # soit n_fft//2 +1 = (samplingFrequency*dt/k_freq)//2 + 1
        num_spectrograms_bins = spectrogram.T.shape[-1]

        linear_to_mel_weight_matrix = librosa.filters.mel(
            sr=self.samplingFrequency,
            #n_fft=int(dt*self.samplingFrequency) + 1,
            n_fft=int(dt*self.samplingFrequency/k_freq) + 1,
            n_mels=num_spectrograms_bins).T

        mel_spectrogram = np.tensordot(
            spectrogram.T,
            linear_to_mel_weight_matrix,
            1)

        return np.log(mel_spectrogram + 1e-6)

    def plotSpectrogram(self, dt=0.025):

        spectrogram = self.spectrogram(dt)

        sns.heatmap(np.rot90(spectrogram.T), cmap='inferno',
                    vmin=0, vmax=np.max(spectrogram)/3)
        loc, labels = plt.xticks()
        l = np.round((loc-loc.min())*self.length /
                     self.samplingFrequency/loc.max(), 2)
        plt.xticks(loc, l)
        loc, labels = plt.yticks()
        l = np.array(loc[::-1]*self.samplingFrequency/2/loc.max(), dtype=int)
        plt.yticks(loc, l)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

    def plotLogMelSpectrogram(self, dt=0.025):

        logMelSpectrogram = self.logMelSpectrogram(dt)
        sns.heatmap(np.rot90(logMelSpectrogram),
                    cmap='inferno', vmin=-6)
        loc, labels = plt.xticks()
        l = np.round((loc-loc.min())*self.length /
                     self.samplingFrequency/loc.max(), 2)
        plt.xticks(loc, l)
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Mel)")

    def normalizeLength(self, duration=1):

        normalizedSamples = duration * self.samplingFrequency

        if self.length >= normalizedSamples:
            self.audioSignal = self.audioSignal[:int(normalizedSamples)]
            self.length = normalizedSamples

        else:
            self.audioSignal = np.concatenate(
                [self.audioSignal, np.zeros(int(normalizedSamples - self.length))])
            self.length = normalizedSamples
        return self

    def addWhiteNoise(self, amplitude=0.05):

        whiteNoise = np.random.normal(
            0, amplitude*np.max(np.abs(self.audioSignal)), self.length)
        self.audioSignal = np.array(self.audioSignal + whiteNoise)

        return self


alphabet = [chars for chars in " ABCDEFGHIJKLMNOPQRSTUVWXYZ'"]
character_encoder = keras.layers.StringLookup(
    vocabulary=alphabet, oov_token="")
character_decoder = keras.layers.StringLookup(
    vocabulary=character_encoder.get_vocabulary(), oov_token="", invert=True)


def decode_batch_predictions(pred):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Greedy search : décodage le plus rapide, ne mène pas forcément au texte le plus probable
    results = keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True)[0][0]

    # on itère sur la prédiction et on récupère le texte
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(
            character_decoder(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


def CTC_loss(y_test, y_pred):

    batch_len = tf.cast(tf.shape(y_test)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_test)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(
        y_test, y_pred, input_length, label_length)

    return loss


def predict(model, filePath):
    logMelSpectrogram = audioFile(filePath, normalize=True, sr=16000).normalizeLength(17) \
        .logMelSpectrogram(k_temp=.7, k_freq=1.5)
    logMelSpectrogram = np.array([(logMelSpectrogram)])
    pred = "prediction: " + \
        decode_batch_predictions(model.predict(logMelSpectrogram))[0]
    st.write(pred)
    return



pages = ["Demo", "CTC algorithm", "Dataset", "Model", "Results"]

page = st.sidebar.radio("navigate", pages)

if page==pages[0]:
    st.title('Speech Recognition')
    st.write("Try for yourself and laugh at the model's efforts to transcribe what you said")
    st.write("It's far from perfect but it's a homemade, end-to-end model. Not some pre-trained thing!")
    st.write("ChatGPT, we're coming for you")
    st.write("Credit to [stefanrmmr](https://github.com/stefanrmmr/streamlit_audio_recorder) for the streamlit audio recorder")
 
    custom_objects = {"CTC_loss": CTC_loss}
    with keras.utils.custom_object_scope(custom_objects):
        model5 = keras.models.load_model('model/model.h5')
    

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    val = st_audiorec()
 
    if isinstance(val, dict):  # retrieve audio data
        with st.spinner('retrieving audio-recording...'):
            ind, val = zip(*val['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            val = np.array(val)             # convert to np array
            sorted_ints = val[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            wav_bytes = stream.read()

        # wav_bytes contains audio data in format to be further processed
        # display audio data as received on the Python side
        #st.audio(wav_bytes, format='audio/wav')

        audioSignal = np.asarray(np.frombuffer(stream.getbuffer(), dtype = "int32"), dtype = np.float64)
        signalPredict(model5, audioSignal, 16000, 17)





if page==pages[1]:
    st.title("CTC algorithm")

    col1, col2 = st.columns([3, 6])
    
    with col1:
        st.markdown('#')
        st.markdown('#')
        alignement = st.checkbox("The alignment problem")
        if alignement:
            st.write("- We have audio records, and their transcriptions, but no way to align both...")
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
      

        CTC = st.checkbox("A solution: CTC algorithm")     
        if CTC:
            st.write("""
                     - Assign a probability to each alphabet's character per time-step 
                     - Maximize the probability of all alignments consistent with the label targeted
                     - Predictions' length are capped by the the number of time-steps
                     
                     Want to know more? [Here](https://distill.pub/2017/ctc/) is a much better explanation.
                     And [here](https://www.cs.toronto.edu/~graves/icml_2006.pdf) is the original paper.
                     """)
    
    with col2:
        st.image('images/spectro.jpg', width = 400)
        cats = ['images/script1.jpg', 'images/script2.jpg', 'images/script3.jpg', 'images/script4.jpg', 'images/script5.jpg']
        placeholder = st.empty()
        k=0
        if alignement:
            while(not CTC):
                rank = int(k%5)
                placeholder.image(cats[rank], width = 400)
                k+=1
                time.sleep(1)
        if CTC:
            st.image('images/scripts.jpg', width = 400)
        
    
if page==pages[2]:
    
    st.title('Dataset')

    st.write('We used a (too) small extract of the [LibriSpeech ASR corpus](https://www.openslr.org/12) to train our model')    
    
    st.write("""
             Working on 30+ k (7 Go / 100+ h) of records and their transcripts, we need to:
                        
            - normalize the audio duration
            
            - maintain consistency between transcript's length and our choice of time-step
             """)
    
    with open('data/train_metadata_full', 'rb') as f:
        df_train = pickle.load(f)
    with open('data/test_metadata_full', 'rb') as f:
        df_test = pickle.load(f)
        
    rootPathTrain = '/content/drive/My Drive/LibriSpeech/train/'
    rootPathTest = '/content/drive/My Drive/LibriSpeech/test/'
    
    df_full = pd.concat([df_train, df_test], axis = 0)
    
    col1, col2 = st.columns([4, 1])
    
    with col2:
        maxDuration = st.slider("duration (s)", min_value = 0.0, max_value = 25.0, 
                            step = .1, value = 10.0)
        timeStep = st.slider("time step (ms)", min_value = 0.0, max_value = 30.0, step = .5, 
                         value = 30.0)

    with col1:
       
        fig = plt.figure(figsize= (15, 10))
        plt.scatter(df_full[df_full['rootPath']==rootPathTrain]['audio duration'], df_full[df_full['rootPath']==rootPathTrain]['label length'], c = 'orange', label = 'train files')
        plt.scatter(df_full[df_full['rootPath']==rootPathTest]['audio duration'], df_full[df_full['rootPath']==rootPathTest]['label length'], c = 'green', label = 'test files')
        plt.legend(fontsize = 12)
        plt.xlim(left = 0, right = 35)
        plt.ylim(bottom = 0, top = 530)
    
    
        plt.vlines(x=maxDuration, ymin = 0, ymax = 530, colors = 'black')
        plt.hlines(y=maxDuration/(.001*timeStep)/2, xmin = 0, xmax = 35, colors = 'black')
        plt.xlabel('audio duration (s)', fontsize = 12)
        plt.ylabel('label length (nb char)', fontsize = 12)
        plt.title('Label length vs audio duration', fontsize = 18);
        
        
        plt.xlabel('audio duration (s)', fontsize = 12)
        plt.ylabel('label length (nb char)', fontsize = 12)
        st.pyplot(fig)

if page==pages[3]:
    
    st.title('Model')
    st.write("We used a structure directly derived from [Deep Speech 2](https://arxiv.org/abs/1512.02595)")
    st.write("And we could never have made it without [this invaluable resource](https://keras.io/examples/audio/ctc_asr/) ")
    st.markdown("##")
    
    col1, col2, col3 = st.columns(3)

    with col1:    
        couchesConvolution = st.checkbox('Convolutional layers')
    with col2:
        couchesRNN = st.checkbox('RNN layers')
    with col3:
        output = st.checkbox('output')
        
    st.image('images/model0.jpg', width = 600)
    
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()
    
    if couchesConvolution:
        st.image('images/model1.jpg', width = 600)
    if couchesRNN:
        st.image('images/model2.jpg', width = 600)
    if output:
        st.image('images/model3.jpg', width = 600)

    
if page==pages[4]:
    
    with open('data/bigTestResults', 'rb') as f:
        df_results = pickle.load(f)
    
    df_results['localPath'] = 'test/' + df_results['fileDirectory'] + df_results['fileName']
    
    st.title('Results')
    
    st.header('Selecting our champion')
    

    with open('data/history0_valLoss.pickle', 'rb') as f:
        history0 = pickle.load(f)
    with open('data/history12_valLoss.pickle', 'rb') as f:
        history12 = pickle.load(f)
    history12big = [80.67, 61.40, 53.97, 49.64, 47.09, 45.89, 44.51, 43.65, 43.37]
    
    precision_v12big = [df_results['v12big_' + str(k)+ '_d'].mean() for k in range(1, 10)] 
    
    with st.expander("CTC loss  / Accuracy"):
    
        st.write("""Starting from a baseline configuration, we tried several options on 
                 12 epochs, 800 train / 200 test samples (approx 1h under Colab premium GPU).
                 Then our champion was trained on 9 epochs over 28 000 train / 2500 test samples.
                 Finally, we use Levenshtein's distance to measure accuracy.
                 """)
        
        col1, col2, col3, col4 = st.columns(4, gap = "small")
        with col1:    
            baseline = st.checkbox('V0 Baseline validation loss - 800 train / 200 test')
        with col2:
            v12 = st.checkbox('V12 Champion validation loss - 800 train / 200 test')
        with col3:
            v12b = st.checkbox('V12 Champion validation loss - 28 000 train / 2 500 test')
        with col4:
            v12b_p = st.checkbox('V12 Champion precision - 28 000 train / 2 500 test')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plt.figure()
            if baseline:
                plt.plot(history0, color = 'black', label = 'baseline')
            if v12:
                plt.plot(history12, color = 'grey', label = 'champion')
            if v12b:
                plt.plot(history12big, color = 'orange', label = 'champion - full training')
            plt.legend(fontsize = 10)
            plt.xlabel('epochs')
            plt.ylabel('CTC loss')
            plt.title("CTC loss on validation data", fontsize = 10)
            st.pyplot(fig)
        
        with col2:
    
            fig = plt.figure()
            if v12b_p:
                plt.plot(precision_v12big, color = 'orange', label = 'champion - full training - greedy decoding')
                plt.scatter([8], [0.874], color = 'orange', marker = '*', label ='beam search decoding')
            plt.legend(fontsize = 10)
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.title('Accuracy on validation data', fontsize = 10)
            st.pyplot(fig)

    st.header('Predictions on validation dataset')
    st.write("""
             And here are the predictions of our fully trained champion!
             \nWe display them by groups of 3, randomly selected within our 2500 samples validation dataset.
             \nYou can compare 2 different epoch's predictions: last one's not always the best
             \n9 vs 9_2 show epoch 9 with greedy decoding (as all others) vs beam search decoding. 
             """)
    epochs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '9_2']
    selection = st.multiselect("select 2 epochs displayed", epochs, 
                               default = ['5', '9'], max_selections = 2)          
    st.button("random select", on_click = 
              randomDisplay(df_results, selection[0], selection[1]))





