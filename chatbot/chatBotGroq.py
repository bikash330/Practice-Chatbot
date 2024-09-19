from langchain_groq import ChatGroq #helps in interacting with llm models 
from langchain_core.prompts import ChatPromptTemplate #helps in prompting chats
from langchain_core.output_parsers import StrOutputParser #helps in display the result comming from llm

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

## calling environment variables

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

#langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## creating caht-bot

prompt = ChatPromptTemplate.from_messages(
    {
    ("system", "You are a helpful assistant. Please provide a response to the user's input."),
    ("user", "Question: {question}")
    }
)

#steamlit 

st.title("Practice example with llama API")
input_txt = st.text_input("Search the topic")


# open ai llm call
model = ChatGroq(model="llama3-8b-8192")
output = StrOutputParser()

# chain

chain = prompt|model|output

if input_txt:
    st.write(chain.invoke({'question':input_txt}))

