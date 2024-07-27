import os
import langchain
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0, separators=["\n",'.  '])
loader = TextLoader('data.txt')
docs = loader.load()
split_docs = text_splitter.split_documents(docs)



split_docss = [Document(page_content=t.page_content, metadata=t.metadata) for t in split_docs]
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docss,embeddings)

os.environ['GROQ_API_KEY']='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'
retriever=vectorstore.as_retriever()
retriever.search_kwargs['k'] = 3

st.set_page_config(page_title="Lloyds Mobile Rewards Section", layout="wide")

st.title("Mock Rewards Section")
st.header("Welcome to Your Rewards")
st.write("Find exclusive offers and discounts just for you!")

st.sidebar.title("Check your eligibility to turn on the discover mode!")
# category = st.sidebar.selectbox("Select Category", ["All", "Electronics", "Food", "Groceries"])