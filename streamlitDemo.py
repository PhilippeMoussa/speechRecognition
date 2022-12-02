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



pages = ["CTC loss", "Exploration du dataset", "Modèle", "Pipeline", 
         "Résultats", "Conclusion et perspectives"]

page = st.sidebar.radio("select", pages)

if page==pages[0]:
    st.title("CTC loss")

    col1, col2 = st.columns([3, 6])
    
    with col1:
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
        alignement = st.checkbox("alignement")
        if alignement:
            st.write("- Plutôt que de prédire 'le' bon alignement...")
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
        
        CTC = st.checkbox("CTC: principe")     
        if CTC:
            st.write("""
                     - ...Assigner une probabilité à chaque caractère par time-step
                     - Maximiser la probabilité des alignements compatibles avec la transcription
                     - Une contrainte: longueur de prédiction inférieure ou égale au nombre de time-steps
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
    
    st.title('Exploration du dataset')
    st.header('Dataset LibriSpeech - langue anglaise')    
    
    st.write("""
             Nous disposons de plus de 30 000 enregistrements audio (7 Go / 100h) en langue anglaise 
             et de leurs transcriptions. Il nous faut :
                 
            - normaliser la durée des enregistrements
            
            - maintenir la compatibilité entre la longueur des étiquettes et la résolution temporelle utilisée
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
    
    st.title('Modèle')
    st.subheader("Structure dérivée de Deep Speech 2")
    
    st.markdown("##")
    
    col1, col2, col3 = st.columns(3)

    with col1:    
        couchesConvolution = st.checkbox('Couches convolution')
    with col2:
        couchesRNN = st.checkbox('Couches RNN')
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
    
    st.title('Pipeline de données')

    st.markdown("##")
    
    col1, col2, col3 = st.columns(3, gap = "small")

    with col1:    
        preprocessing = st.checkbox('Preprocessing')
    with col2:
        augmentation = st.checkbox('Augmentation')
    with col3:
        modele = st.checkbox('Modèle')
    
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
    
    st.title('Résultats')
    
    st.header('Sélection de configuration')
    

    with open('data/history0_valLoss.pickle', 'rb') as f:
        history0 = pickle.load(f)
    with open('data/history12_valLoss.pickle', 'rb') as f:
        history12 = pickle.load(f)
    history12big = [80.67, 61.40, 53.97, 49.64, 47.09, 45.89, 44.51, 43.65, 43.37]
    
    precision_v12big = [df_results['v12big_' + str(k)+ '_d'].mean() for k in range(1, 10)] 
    
    with st.expander("CTC loss  / Précision"):
    
        st.write("""A partir d'une configuration de base, 
                 nous avons testé plusieurs options en boucle courte: 
                 12 epochs, 800 train / 200 test.
                 Puis le 'champion' ainsi sélectionné a été entraîné
                 sur 9 epochs, 28 000 train / 2500 test.
                 Nous appliquons enfin une métrique basée sur la distance de Levenshtein.
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
            plt.title('Precision on validation data', fontsize = 10)
            st.pyplot(fig)

    st.header('Prédictions sur données de validation')
    st.write("""
             Et voici les prédictions de notre 'champion' après un entraînement long!
             \nNous les affichons par groupe de trois enregistrements, avec leurs labels, 
             sélectionnés aléatoirement
             dans notre dataset de validation (2500 enregistrements).
             \nVous pouvez comparer les résultats obtenus à deux epochs différents.
             \nEn sélectionnant 9 et 9_2, vous pouvez aussi comparer les prédictions de l'epoch 9,
             décodées en mode "greedy' (9) ou avec un "ctc beam search" (9_2).
             """)
    epochs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '9_2']
    selection = st.multiselect("select 2 epochs displayed", epochs, 
                               default = ['5', '9'], max_selections = 2)          
    st.button("random select", on_click = 
              randomDisplay(df_results, selection[0], selection[1]))
   
if page==pages[5]:
    
    st.title('Conclusion et perspectives')

    st.markdown("###")
    st.write("""
             Tous nos remerciements à DataScientest et plus particulièrement à 
             notre chef de cohorte Romain Godet et à notre mentor Paul Lestrat!
             """)
    st.markdown("###")

    st.header("Quelques pistes")

    st.markdown("###")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#")
            st.write("""
            - (Beaucoup) plus de données d'entraînement
                - Deep Speech 2: plus de 11 000h audio! 
            - Autres augmentations de données:
                - stretching, pitch shifting
                - autres bruits
            - Autres configurations pre-processing/augmentation/modèle
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
                     - Meilleur décodage des prédictions
                     - Métrique: mesure de la qualité des prédictions
                     """)
        with col2:
            st.image('images/MLmetrics.jpg', width = 400)

    st.markdown("##")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
                     - Modèle de langage
                     """)
        with col2:
            st.image("images/languageModel.png", width = 400)
  
    st.markdown("##")
  
    with st.container():
         st.write("...")
                 
