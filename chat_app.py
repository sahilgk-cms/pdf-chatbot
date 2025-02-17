import streamlit as st
import sys
import os
from constants import *
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.chat import *
from utils.vector_embeddings import *


st.title("Chatbot-for BAT and OAK")

st.markdown('''
            This is the home page.\n
            There are four pages with two versions for each - BAT and OAK.\n
            **Page 1:** BAT using OpenAI Embeddings & LLM.\n
            **Page 2:** BAT using Gemini & basic prompt template framework.\n
            **Page 3:** OAK using OpenAI Embeddings & LLM.\n
            **Page 4:** OAK using Gemini & basic prompt template framework.
''')
