# pages/summarization.py

import requests
import streamlit as st

from src.utils.constants import SUMMARIZER_API_URL

st.title("Text Summarization")

input_text = st.text_area("Enter text to summarize", height=200)

if st.button("Summarize"):
    if input_text.strip():
        # Ensure the payload is in the correct format
        payload = {"input_text": input_text}
        
        response = requests.post(SUMMARIZER_API_URL, json=payload)
        if response.status_code == 200:
            summary = response.json().get("summary", "")
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.error(f"Error: {response.status_code}. {response.text}")
    else:
        st.error("Please enter some text.")
