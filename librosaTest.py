#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:39:41 2022

@author: audeetphilippemoussa
"""

import streamlit as st
import librosa
st.set_page_config(layout="wide")
import time


signal, freq = librosa.load('learn.m4a')
st.write("frequency:" + str(freq))

