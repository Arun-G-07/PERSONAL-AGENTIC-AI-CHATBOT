import os
from dotenv import load_dotenv
load_dotenv()


#SETTING UP API KEYS
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TRAVILY_API_KEY")  #For web search/ internet search or calls
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

#Importing necessary libraries and packages

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


#Setting up llms
openai_llm = ChatOpenAI(model="gpt-4o-mini")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")


#Setting up tools
search_tool = TavilySearchResults(max_results=2)


#Setting AGENTS
from langgraph.prebuilt import create_react_agent  #ReAct - reaoning then acting
from langchain_core.messages.ai import AIMessage



system_prompt = "Act as an AI Chatbot who is smart and friendly"


# agent = create_react_agent(
#     model=groq_llm,
#     tools=[search_tool],
#     prompt=system_prompt #role of the agent

# )

# query = "Tell me about the trends in crypto markets"
# state = {"messages": query}
# response=agent.invoke(state)
# messages = response.get("messages")
# #The response returns many values, first we separete messages from that, and it consists of both 
# #human as well as ai-messages, next step is to filter out only ai_messages
# ai_messages = [message.content for message in messages if isinstance(message,AIMessage)]
# print(ai_messages[-1]) #printing the last message

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id)

    tools=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]