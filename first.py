import os
from dotenv import load_dotenv
load_dotenv()
os.getenv('HF_TOKEN')

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

llm=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))

loader = TextLoader('Text.txt')
docs=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
split_docs = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


query = "What are the side effects"
retrieved_docs = retriever.invoke(query)

pairs = [(query, doc.page_content) for doc in retrieved_docs]
scores = reranker.predict(pairs)
ranked = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
top_docs = [retrieved_doc for score, retrieved_doc in ranked[:5]]

context = "\n".join([doc.page_content for doc in top_docs])

prompt = f"""
     Answer the questions based on the context below.
     Context: {context}
     Question: {query}
"""

response = llm.invoke(prompt)
print(response.content)