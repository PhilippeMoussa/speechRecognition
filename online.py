#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:45:57 2022

@author: audeetphilippemoussa
"""

from pyngrok import ngrok 
!ngrok authtoken 2HiQFjnpvwXLEZYimSRMK44umZQ_rnXyKJ31jvu7ABMZKHxq
!nohup streamlit run app.py & 

url = ngrok.connect(port = 8501)
url #generates our URL

!streamlit run --server.port 80 streamlit_sandbox.py >/dev/null #used for starting our server