import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def get_groqai_response(input_txt):
    response = requests.post("http://localhost:8000/disease/invoke",json={"input":{"topic":input_txt}})
    return response.json()["output"]["content"]

## streamlit framework

st.title("Chat with me for any disease realted topic")
input_txt = st.text_input("Disease you want to know about")

if input_txt:
    st.write(get_groqai_response(input_txt))

