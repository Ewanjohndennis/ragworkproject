import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain.agents import create_agent

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)
chat_model = ChatHuggingFace(llm=llm)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wiki_tool]

app = create_agent(
    model=chat_model,
    tools=tools,
    system_prompt="You are a helpful assistant that uses Wikipedia to verify facts."
)

query = "Who is the current Prime Minister of the UK and what is their age?"

result = app.invoke({"messages": [("user", query)]})

final_message = result["messages"][-1]
print(final_message.content)