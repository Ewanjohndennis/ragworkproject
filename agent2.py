import os
import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

# 1. Environment & Token Setup
load_dotenv()
# This pulls HF_TOKEN from your .env file
token = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

# 2. Model Setup
endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=token
)
llm = ChatHuggingFace(llm=endpoint)

# 3. Tools Definition
def search_ddgo(query: str):
    """Search the web using DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

def add(a: float, b: float):
    """Add two numbers together."""
    return a + b

def multiply(a: float, b: float):
    """Multiply two numbers together."""
    return a * b

# 4. Agent Creation
math_agent = create_react_agent(
    model=llm,
    tools=[add, multiply]
)
research_agent = create_react_agent(
    model=llm,
    tools=[search_ddgo]
)

# 5. Graph State & Nodes
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_agent: str

def agent_node(state, agent, name):
    # Pass only the messages to the agent
    result = agent.invoke({"messages": state["messages"]})
    
    # In the 2026 version of create_react_agent, result is a dict with 'messages'
    new_messages = result["messages"]
    
    return {
        "messages": new_messages,
        "current_agent": "supervisor",
    }

def supervisor_node(state: AgentState):
    last_message = state["messages"][-1]
    
    if isinstance(last_message, dict):
        content = last_message["content"].lower()
    else:
        content = last_message.content.lower()

    # Logic to stop if math is done
    if "the result is" in content or "final answer" in content:
        return {"current_agent": "FINISH"}

    if any(keyword in content for keyword in ["multiply", "add", "math"]):
        return {"current_agent": "math_agent"}

    elif any(keyword in content for keyword in ["temperature", "what is", "research", "delhi"]):
        return {"current_agent": "res_agent"}

    else:
        # Default back to research if unsure
        return {"current_agent": "res_agent"}

# 6. Building the Workflow
workflow = StateGraph(AgentState)

workflow.add_node("res_agent", lambda state: agent_node(state, research_agent, "res_agent"))
workflow.add_node("math_agent", lambda state: agent_node(state, math_agent, "math_agent"))
workflow.add_node("supervisor", supervisor_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["current_agent"],
    {
        "res_agent": "res_agent",
        "math_agent": "math_agent",
        "FINISH": END
    }
)

workflow.add_edge("res_agent", "supervisor")
workflow.add_edge("math_agent", "supervisor")

app = workflow.compile()

# 7. Execution
if __name__ == "__main__":
    query = "what is the temperature in Delhi ? Multiply the temperature by 2 and then add 3 to the result"
    
    print("--- Starting Multi-Agent Task ---")
    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "current_agent": "supervisor"
    })
    
    print("\n--- Final Message ---")
    print(result['messages'][-1].content)