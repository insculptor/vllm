import requests
import streamlit as st

# Define the FastAPI URL (adjust this if the port or URL differs)
API_URL = "http://localhost:8000/generate"

# Set up the Streamlit app layout
st.title("vLLM Inference App")

st.write("""
### Enter Text to Generate Inference:
""")

# Create a text input form in Streamlit
input_text = st.text_area("Input Text", placeholder="Enter some text for the model to complete...")

# Define optional parameters (max_tokens, temperature, etc.)
max_tokens = st.number_input("Max Tokens", min_value=1, max_value=2048, value=1024)
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8)
top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9)

# Add a stream toggle
stream = st.checkbox("Stream Output", value=True)

# Add a submit button
if st.button("Generate"):
    # Ensure the input text is not empty
    if not input_text.strip():
        st.error("Please enter some text for the model.")
    else:
        # Prepare the payload for the API request
        payload = {
            "prompt": input_text,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": stream  # Use the value from the checkbox
        }

        # If streaming is enabled, handle the streaming response
        if stream:
            try:
                with requests.post(API_URL, json=payload, stream=True) as response:
                    if response.status_code == 200:
                        st.write("### Generated Text (Streaming):")
                        # Display streamed chunks as they arrive
                        for chunk in response.iter_content(chunk_size=None):
                            if chunk:
                                data = chunk.decode('utf-8').strip("\0")  # Handle stream delimiter
                                st.write(data)  # Display chunk in UI
                    else:
                        st.error(f"Error: {response.status_code}. Could not generate text.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            # If streaming is disabled, handle it as a normal request
            try:
                response = requests.post(API_URL, json=payload)

                # Check for a successful response
                if response.status_code == 200:
                    result = response.json()
                    st.write("### Generated Text:")
                    # Display the generated text
                    st.success("\n".join(result['text']))
                else:
                    st.error(f"Error: {response.status_code}. Could not generate text.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
