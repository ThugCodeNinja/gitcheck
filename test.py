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
for i in response:
  print(i.page_content)
     
st.set_page_config(page_title="Lloyds Mobile Rewards Section", layout="wide")

st.title("Mock Rewards Section")
st.header("Welcome to Your Rewards")
st.write("Find exclusive offers and discounts just for you!")

st.sidebar.title("Check your eligibility to turn on the discover mode!")
# category = st.sidebar.selectbox("Select Category", ["All", "Electronics", "Food", "Groceries"])
if 'count' not in st.session_state:
    st.session_state.count = 0

enable_controls = st.session_state.count == 5

def increment_count():
    st.session_state.count += 1

st.sidebar.title(f'Number of transactions : {st.session_state.count}')
# st.write(f"Button has been pressed {st.session_state.count} times")
button=st.sidebar.button("Mock Transaction")
if button:
    increment_count()

# Display the updated count
if enable_controls:
    st.sidebar.write("You are now eligible for discover mode ")
    user_input = st.sidebar.text_input("Enter your request or choice of category", key='user_input')
    print(user_input)
    response=retriever.get_relevant_documents(query=st.session_state.user_input)

if not enable_controls:
    st.sidebar.write(f"Keep going {20-st.session_state.count} times.")
