import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import date, datetime, timedelta
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)



# Function to calculate exact dates based on user input
def calculate_date(date_string, current_date):
    current_date = datetime.strptime(current_date, "%Y-%m-%d")
    
    # Handle relative days like "three days"
    if "day" in date_string.lower():
        try:
            num_days = int(date_string.split()[0])
            return (current_date + timedelta(days=num_days)).strftime("%Y-%m-%d")
        except:
            return "Unable to calculate date"
    
    # Handle "next Monday"
    if "next" in date_string.lower():
        day = date_string.lower().split()[-1]
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        target_day = days.index(day)
        days_ahead = target_day - current_date.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (current_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    # Handle "tomorrow"
    elif "tomorrow" in date_string.lower():
        return (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    return "Unable to calculate date"

def create_booking_agent(llm):
    # Define a Tool for date calculation
    date_tool = Tool(
        name="DateCalculator",
        func=calculate_date,
        description="Calculates the date based on the input string and current date"
    )

    tools = [date_tool, TavilySearchResults(max_results=1)]

# Updated Prompt Template with current_date and required variables
    prompt_template: str = """
You are an AI assistant specialized in booking appointments. Your task is to extract a valid date from the user's input and convert it into the format YYYY-MM-DD.

You are expected to handle:
- Relative dates like "tomorrow", "next Monday", "three days from now", "in two weeks".
- Absolute dates like "15th September 2024", "October 10", or "22/10/2024".
- Ambiguous phrases like "later this week", "next weekend", or "early next month" by estimating the most probable date based on the current date.
- If the sentence contains multiple date references, choose the most relevant one based on context.

Today's date: {current_date}

Available tools: {tool_names}

Tool details:
{tools}

You should follow this process:
1. Thought: Analyze the sentence to understand the context and identify the phrase related to the date.
2. Action: Convert the date or phrase into the format YYYY-MM-DD.
3. If it's a relative date, calculate it based on today’s date.
4. If the date is unclear, make an educated guess and document your reasoning.
5. Return the exact date in YYYY-MM-DD format.

Current conversation:
Human: {input}

Use the following format to present your result:
- Thought: Describe how you are understanding the input and processing it.
- Final Answer: The appointment date in YYYY-MM-DD format.

Some examples of how you should process different types of input:
- Input: "I need an appointment for next Monday."
  Thought: The user wants an appointment on the Monday of the next week. 
  Final Answer: [YYYY-MM-DD for next Monday]
  
- Input: "Schedule the meeting for tomorrow."
  Thought: The user is referring to tomorrow.
  Final Answer: [YYYY-MM-DD for tomorrow]

- Input: "Set the reminder for three days from now."
  Thought: The user wants something scheduled three days from today.
  Final Answer: [YYYY-MM-DD for three days later]

- Input: "Book a trip on 22nd October 2024."
  Thought: The user provided a specific date.
  Final Answer: 2024-10-22.

{agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template=prompt_template)

# Construct the ReAct agent by passing the required 'tools' and 'tool_names'
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    return agent_executor

def booking_appointment(booking_time):
    agent_executor = create_booking_agent(llm)
    current_date = date.today().strftime("%Y-%m-%d")
    
    result = agent_executor.invoke({
        "input": booking_time,
        "current_date": current_date
    })

    return result['output']

def checking_appointment(chat):
    list_words = ["book", "booking", "reserve", "appointment"]
    for word in list_words:
        if word in chat.lower().split():
            return True


def main():
    
    st.header("Booking chatbot")
    st.write("This chatbot is designed to assist with booking appointments.")
    chat = st.text_input("write any thing ")
    if checking_appointment(chat) is True:
        appointment_date = booking_appointment(chat)
        st.write(f"Your booking has been registered for {appointment_date}")

if __name__ == "__main__":
    main()
