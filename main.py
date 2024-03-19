import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Streamlit UI setup
st.set_page_config(page_title='Chat with Files')
st.title('Chat with Files')

# File upload
uploaded_file = st.file_uploader('Upload a PDF document', type='pdf')
# Query text
query_text = st.text_input('Enter your question:', placeholder='Ask me anything about the document.', disabled=not uploaded_file)