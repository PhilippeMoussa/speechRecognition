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
import time
import random
#import tensorflow as tf

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
     
### fin des utilitaires pour demo



pages = ["CTC loss", "Dataset exploration", "Model", "Pipeline", 
         "Results", "Conclusion"]

page = st.sidebar.radio("navigate", pages)

if page==pages[0]:
    st.title("CTC loss")

    col1, col2 = st.columns([3, 6])
    
    with col1:
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
        alignement = st.checkbox("alignment")
        if alignement:
            st.write("- Instead of predicting 'the' right aligment...")
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
        
        CTC = st.checkbox("CTC: principle")     
        if CTC:
            st.write("""
                     - ...assign a probability to each alphabet's character per time-step 
                     - Maximize the probability of all alignments consistent with the label targeted
                     - One constraint: predictions' length are capped by the the number of time-steps
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
        
    
if page==pages[1]:
    
    st.title('Dataset exploration')
    st.header('LibriSpeech extract - English')    
    
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

if page==pages[2]:
    
    st.title('Model')
    st.subheader("Building on a structure derived from Deep Speech 2 ")
    st.markdown("##")
    
    col1, col2, col3 = st.columns(3)

    with col1:    
        couchesConvolution = st.checkbox('Convolutional layers')
    with col2:
        couchesRNN = st.checkbox('RNN layers')
    with col3:
        output = st.checkbox('output')
        
    st.image('images/model0.jpg')
    
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()
    
    if couchesConvolution:
        st.image('images/model1.jpg')
    if couchesRNN:
        st.image('images/model2.jpg')
    if output:
        st.image('images/model3.jpg')


if page==pages[3]:
    
    st.title('Data pipeline')

    st.markdown("##")
    
    col1, col2, col3 = st.columns(3, gap = "small")

    with col1:    
        preprocessing = st.checkbox('Preprocessing')
    with col2:
        augmentation = st.checkbox('Augmentation')
    with col3:
        modele = st.checkbox('Model')
    
    st.write("\n")

    st.image('images/pipeline0.jpg')
    
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()
    placeholder4 = st.empty()
    
    if preprocessing:
        placeholder1.image('images/pipeline1.jpg')
    if augmentation:
        placeholder2.image('images/pipeline2.jpg')
    if modele:
        placeholder3.image('images/pipeline3.jpg')
        placeholder4.image('images/pipeline4.jpg')

    
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
    
    with st.expander("CTC loss  / Précision"):
    
        st.write("""Starting from a baseline, we tried several options on 
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
   
if page==pages[5]:
    
    st.title('Conclusion ')

    st.markdown("###")
    st.write("""
             Our thanks to DataScientest and our mentor Paul Lestrat!
             """)
    st.markdown("###")

    st.header("A few follow-up ideas")

    st.markdown("###")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#")
            st.write("""
            - (Much) more training data
                - Deep Speech 2: 11 000h records! 
            - Other data augmentation
                - stretching, pitch shifting
                - real life noises
            - Other configurations pre-processing/augmentation/model
            """)
        with col2:
            st.image('images/audioData.jpg', width = 400)
    
    st.markdown("##")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.write("""
                     - Improve decoding of predictions
                     - Metric: move on to word error rate 
                     """)
        with col2:
            st.image('images/MLmetrics.jpg', width = 400)

    st.markdown("##")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
                     - Language model: lexicon, n-grams
                     """)
        with col2:
            st.image("images/languageModel.png", width = 400)
  
    st.markdown("##")
  
    with st.container():
         st.write("...")
                 
