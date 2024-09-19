from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq 

from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="Practice Chatbot",
    version="1.0",
    description="Simple API Srever for GROQ LLMs"
)

add_routes(
    app,
    ChatGroq(),
    path="/groqai"
)

model = ChatGroq()

prompt = ChatPromptTemplate.from_template("Summarize the causes and risk factors for {topic}. Create 3 concise explanations with a maximum of 100 words each")

add_routes(
    app,
    prompt | model,
    path="/disease"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
