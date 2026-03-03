import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf = os.getenv('HF_TOKEN')

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph_supervisor import create_supervisor

# --- UI ---
st.title("Multi-Agent Chat Assistant 🤖💬 (with Temporary Memory)")

# SESSION MEMORY (clears on refresh)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # messages shown in UI
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = []  # messages sent to supervisor

# --- Initialize LLM + Tools ---
endpoint = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b")
llmhg = ChatHuggingFace(llm=endpoint)

search = DuckDuckGoSearchRun()

def subtract(a: int, b: int) -> int:
    "Subtract second number from first number and give result"
    return a - b

def add(a: int, b: int) -> int:
    "Add both the numbers and give the result"
    return a + b

agent = create_agent(
    model=llmhg,
    tools=[add, subtract],
    name="math_agent"
)

agent2 = create_agent(
    model=llmhg,
    tools=[search],
    name="search_agent"
)

Supervisor = create_supervisor(
    model=llmhg,
    agents=[agent, agent2],
    name="supervisor"
)

app = Supervisor.compile()

# --- Display chat history (UI only) ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat input box ---
user_query = st.chat_input("Type your message...")

if user_query:
    # Add user message to UI memory
    st.session_state.chat_history.append(
        {"role": "user", "content": user_query}
    )

    # Add user message to agent memory
    st.session_state.agent_memory.append(
        HumanMessage(content=user_query)
    )

    # Show user bubble
    with st.chat_message("user"):
        st.write(user_query)

    # Process with supervisor agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass full conversation memory
            resp = app.invoke({"messages": st.session_state.agent_memory})

            answer = resp["messages"][-1].content
            st.write(answer)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
    st.session_state.agent_memory.append(
        HumanMessage(content=answer)
    )