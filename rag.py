from typing import TypedDict, List
from langgraph.graph import START, END

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


load_dotenv()


llm = ChatGroq(model="llama-3.3-70b-versatile")

"""
Summaries
"""

raw_docs = [
    Document(page_content="""
    In production AI systems, retrieval quality is often degraded due to noisy embeddings,
    outdated documentation, and poor chunking strategies. Many systems fail when documents
    contain overlapping information across multiple domains, such as finance and healthcare.
    """),
    
    Document(page_content="""
             """)
]