import os, getpass
from dotenv import load_dotenv
load_dotenv()
hf=os.getenv('HF_TOKEN')
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph_supervisor import create_supervisor


endpoint=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b")
llmhg=ChatHuggingFace(llm=endpoint)
search=DuckDuckGoSearchRun()
def subtract(a: int, b:int)-> int:
    "Subtract second number from first number and return the result"
    return a-b
def add(a: int, b:int)-> int:
    "Add second number to first number and return the result"
    return a+b
agent=create_agent(model=llmhg, tools=[add,subtract], name="Math Agent")
agent2=create_agent(model=llmhg, tools=[search], name="Seacrh Agent")
query=input("Enter Query:")
Supervisor=create_supervisor(model=llmhg, agents=[agent, agent2], name="Supervisor")
app =Supervisor.compile()

resp=app.invoke({"messages":[HumanMessage(content=query)]})
print(resp['messages'][-1].content)
