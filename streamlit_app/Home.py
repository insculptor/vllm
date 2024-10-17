# pages/chat_completions.py
import os
import sys

import streamlit as st

sys.path.insert(0, os.getenv('ROOT_DIR'))



# Set up the page layout
st.title("vLLM (OpenAI Compatible) Playground")
st.markdown(
    """
    This is a playground to interact with the vLLM model. 
    """
)