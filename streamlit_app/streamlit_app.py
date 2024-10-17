import os
import sys

sys.path.insert(0, os.getenv('ROOT_DIR'))
import streamlit as st

# Set the title of the main page
st.set_page_config(page_title="vLLM App", layout="wide")

st.title("Welcome to the vLLM App")

st.write("""
This is a multi-page application with the following options:
- **Chat Completions:** Generate chat-based responses using the vLLM model.
- **Embeddings Generator:** Generate vector embeddings for your input text.
Use the sidebar to navigate between pages.
""")
