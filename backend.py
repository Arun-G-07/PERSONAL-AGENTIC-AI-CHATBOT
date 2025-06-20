#Setup Pydantic model(schema validation)
# Mainly used to standardize the communication between two system especially through API calls,
# like in which format/type you are giving data and in which format you are getting/returning data
#And whenever 2 system communicates through API there will be a standardized way to communicate

from pydantic import BaseModel
from typing import List

from dotenv import load_dotenv
load_dotenv()

#data contract
#When an information payload from frontend comes, and if it is not like this type, then we can't do further processing
#So, a validation check is happening here

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

#Setting up AI Agent from FrontEnd Request

#Here we have to create an endpoint where the frontend will send requests
# when the information comes it goes to this endpoint and starts further processes

from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

ALLOWED_MODEL_NAMES = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini"]


app = FastAPI(title="LangGraph AI-Agent")


#Creating a chat endpoint
@app.post('/chat')
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request

    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error":"Invalid model name. Kindly selaect a valid AI Model"}
    
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider


    response = get_response_from_ai_agent(llm_id,query,allow_search,system_prompt,provider)
    return response

#Run app, Explore swagger UI docs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)