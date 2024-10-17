# pages/embeddings_generator.py
import os
import sys

sys.path.insert(0, os.getenv('ROOT_DIR'))
import requests
import streamlit as st

from src.utils.constants import EMBEDDING_API_URL

st.title("Embedding Playground")

# Collect multiple texts for embedding
input_texts = st.text_area("Enter multiple texts (one per line)", "").splitlines()

if st.button("Generate Embeddings"):
    # Ensure input is not empty
    if not input_texts:
        st.error("Please provide at least one text.")
    else:
        # Prepare the request payload
        payload = {"input": input_texts}  # List of strings
        try:
            response = requests.post(EMBEDDING_API_URL, json=payload)

            if response.status_code == 200:
                embeddings = response.json().get("embeddings")
                st.write("Generated Embeddings:")
                st.json(embeddings)
            else:
                st.error(f"Failed to generate embedding: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
