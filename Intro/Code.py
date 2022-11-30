#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:08:54 2022

@author: nicotalon
"""
import streamlit as st
from PIL import Image
from IPython.display import Audio

 
      
st.sidebar.title('Navigation')
pages = ["Reconnaissance vocale", "Preprocessing et visualisation des données", "augmentation des données"]
page = st.sidebar.radio("select", pages)

if page==pages[0]:
    st.title("La reconnaissance vocale")
    image = Image.open('/Users/nicotalon/image1.png')
    st.header("Fonctionnement de la reconnaissance vocale")
    st.image(image)

    st.write("""
             - Etude de 2 modèles : 
                 - Modèle de classification
                 - Modèle de transcription
             
             """)

if page==pages[1]:
    st.title("Preprocessing")
    image2 = Image.open('/Users/nicotalon/image2.png')
    st.header("Etapes d'obtention d'un spectrogramme")
    st.image(image2)

    st.write("""
         - Transformation de la pression acoustique en signal analogique puis conversion en signal numérique
         - Décomposition du son en ses composantes fréquentielles
         - Division du son en fenêtres temporelles permettant une représentation tridimensionnelles du signal via le spectrogramme
         
         """)
    image3 = Image.open('/Users/nicotalon/image3.png')
    image4 = Image.open('/Users/nicotalon/image4.png')
    st.header("Spectrogramme de MEL")
    st.image(image3)     
    st.image(image4) 
    st.write("""
         - Pour le même fichier audio : 
             - spectrogramme standard (en haut)
             - spectrogramme de MEL (en bas) qui se calque sur la perception humaine des fréquences
         """)    
         
if page==pages[2]:
    st.title("L'augmentation des données")
    option = st.selectbox(
    'Les différents types de transformations du signal audio',
    ('Pas de transformations','Bruit Blanc', 'Time Shifting', 'Time Stretching', 'Pitching'))
    if option == 'Pas de transformations':
        st.audio('/Users/nicotalon/standard.mp3')
        image01 = Image.open('/Users/nicotalon/standard.png')
        st.image(image01)
    if option == 'Bruit Blanc':
        st.audio('/Users/nicotalon/Bruit blanc.mp3')
        image02 = Image.open('/Users/nicotalon/Bruit blanc.png')
        st.image(image02)
    if option == 'Time Shifting':
        st.audio('/Users/nicotalon/shifting.mp3')
        image03 = Image.open('/Users/nicotalon/shifting.png')
        st.image(image03)
    if option == 'Time Stretching':
        st.audio('/Users/nicotalon/stretching.mp3')
        image04 = Image.open('/Users/nicotalon/stretching.png')
        st.image(image04)
    if option == 'Pitching':
        st.audio('/Users/nicotalon/Pitching.mp3')
        image05 = Image.open('/Users/nicotalon/Pitching.png')
        st.image(image05)
    
    
    
    
   

    
    st.header("Les différents types d'augmentations")
    st.write("""
         - Le bruitage : ajout d'un signal parasite
         - Time Stretching : modification de la vitesse du signal 
         - Time Shifting : décalage du signal
         - Pitching : modification des fréquences du signal
             
         """)          
    
         
         