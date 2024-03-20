import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Streamlit UI setup
st.set_page_config(page_title='Chat with Files')
st.title('Chat with Files')

# File upload
uploaded_file = st.file_uploader('Upload a PDF document', type='pdf')
# Query text
query_text = st.text_input('Enter your question:', placeholder='Ask me anything about the document.', disabled=not uploaded_file)

if uploaded_file is not None:
    # Create a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        # Write the content of the uploaded file to the temporary file
        tmpfile.write(uploaded_file.getvalue())
        tmpfile_path = tmpfile.name

    # Initialize the PyPDFLoader with the uploaded file
    loader = PyPDFLoader(tmpfile_path)


   
    documents = loader.load()
    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    st.write(texts)

    os.remove(tmpfile_path)