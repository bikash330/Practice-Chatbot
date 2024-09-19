import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

tools = [TavilySearchResults(max_results=1)]

# Initialize memory for conversation history
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "What is computer"})
