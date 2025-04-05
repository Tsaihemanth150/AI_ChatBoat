from dotenv import load_dotenv

load_dotenv()  ## loading all the environment variables

import streamlit as st
import os
import google.generativeai as genai


genai.configure(api_key="AIzaSyByUjYci4Xv7d5AE6hSO4FCaaWmAqABYWs")

## function to load Gemini Pro model and get repsonses
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])


def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response


##initialize our streamlit app

st.set_page_config(page_title="DevOps Demo")

st.header("DevOps Demo")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")


if submit and input:
    response = get_gemini_response(input)
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", input))
    st.subheader("The Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("DevBoat", chunk.text))
st.subheader("The Chat History is")

for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")