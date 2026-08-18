import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_core.runnables import RunnablePassthrough

from  dotenv import load_dotenv
load_dotenv()
# Document Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Vector Database
from langchain_chroma import Chroma

# Prompt
from langchain_core.prompts import PromptTemplate

loader=PyPDFLoader("Muhammad_Hassan_Resume .pdf")
doc=loader.load()
# print(doc[0].page_content)

splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
chunks=splitter.split_documents(doc)
#print(chunks)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
model= ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

vector_store=Chroma(
    collection_name="pdf",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)
vector_store.add_documents(chunks)

#retriver
retriever = vector_store.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)
query=input("askh any question ")
result=retriever.invoke(query)
# for i,result in enumerate(result):
#     print(f"/n resutl no{i+1}")
#     print(result.page_content)

#
template=PromptTemplate(
template="""
You are a helpful assistant.

Answer the user's question using only the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
"I don't know based on the provided document."

Answer:
""",
input_variables=["context","question"]
)
prompt=template.invoke(
    {"question":query,"context":result}

)

final_result=model.invoke(prompt)
print(final_result)
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

final_result = model.invoke(prompt)
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | template
    | model
    | parser
)
answer = rag_chain.invoke(final_result)

print(answer)