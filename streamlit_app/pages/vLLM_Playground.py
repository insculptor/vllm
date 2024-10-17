# pages/chat_completions.py
import os
import sys

import requests
import streamlit as st

sys.path.insert(0, os.getenv('ROOT_DIR'))
from src.utils.config import ConfigLoader

# Load configuration
config = ConfigLoader()
api_host = config["apiserver"]["host"]
api_port = config["apiserver"]["vllm_port"]
model_identifier = config["engine_args"]["model"]

# Define the API URL for chat completions
API_URL_CHAT = f"http://{api_host}:{api_port}/v1/chat/completions"

# Set up the page layout
st.title("vLLM OpenAI Compatible Playgrund")

st.write("### Enter Text to Generate a Chat Completion:")

# Create a text input field
input_text = st.text_area("User Message", placeholder="Enter your message here...")

# Add optional parameters
max_tokens = st.number_input("Max Tokens", min_value=1, max_value=2048, value=1024)
temperature = st.slider("Temperature", min_value=0.01, max_value=2.0, value=0.8)
top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9)
stream = st.checkbox("Stream Output", value=True)

# Add a submit button
if st.button("Generate"):
    if not input_text.strip():
        st.error("Please enter some text for the model.")
    else:
        # Prepare the payload
        payload = {
            "model": model_identifier,
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": stream
        }

        if stream:
            try:
                with requests.post(API_URL_CHAT, json=payload, stream=True) as response:
                    if response.status_code == 200:
                        st.write("### Generated Text (Streaming):")
                        for chunk in response.iter_content(chunk_size=None):
                            if chunk:
                                data = chunk.decode('utf-8').strip("\0")
                                st.write(data)
                    else:
                        st.error(f"Error: {response.status_code}. Could not generate text.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            try:
                response = requests.post(API_URL_CHAT, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.write("### Generated Text:")
                    st.success(result['choices'][0]['message']['content'])
                else:
                    st.error(f"Error: {response.status_code}. Could not generate text.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
