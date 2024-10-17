# pages/reranker.py
import os
import sys

sys.path.insert(0, os.getenv('ROOT_DIR'))
import requests
import streamlit as st

import src.utils.constants as c

st.title("Reranker")

st.write("Enter a query and list of documents to rerank them:")

query = st.text_input("Query", placeholder="Enter your query here...")
documents = st.text_area(
    "Documents", 
    placeholder="Enter documents separated by newline...", 
    height=200
)

if st.button("Rerank"):
    if not query.strip() or not documents.strip():
        st.error("Please enter both a query and documents.")
    else:
        # Prepare the payload
        document_list = [doc.strip() for doc in documents.split("\n") if doc.strip()]
        payload = {"query": query, "documents": document_list}

        # Make the POST request
        try:
            response = requests.post(c.RERANKER_API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.write("### Reranked Documents:")
                for doc in result["reranked_documents"]:
                    st.write(f"- {doc}")
            else:
                st.error(f"Error: {response.status_code}. {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
