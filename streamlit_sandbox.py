import streamlit as st

import os
import streamlit.components.v1 as components
#from audiorecorder import audiorecorder
#from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
#from aiortc.contrib.media import MediaRecorder



parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
st_audiorec = components.declare_component("st_audiorec", path=build_dir)

st_audiorec()




#st.title("Audio Recorder2")
#audio = audiorecorder("Click to record", "Recording...")

#if len(audio) > 0:
    # To play audio in frontend:
 #   st.audio(audio, sample_rate = 3000)
  #  st.audio(audio, sample_rate = 4000)
   # st.audio(audio, sample_rate = 8000)
    #st.audio(audio, sample_rate = 16000)


    
    # To save audio to a file:
#    wav_file = open("audio.mp3", "wb")
#    wav_file.write(audio.tobytes())


