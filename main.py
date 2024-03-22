import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Streamlit UI setup
st.set_page_config(page_title='Chat with Files')
st.title('Chat with Files')

# Initialize or get the conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader('Upload a PDF document', type='pdf')

# Function to handle the chat interaction
def handle_chat():
    if 'llm_response' in st.session_state and st.session_state.llm_response:
        # Append the user's question and LLM's response to the history
        st.session_state.history.append(('You', query_text))
        st.session_state.history.append(('Bot', st.session_state.llm_response))
        st.session_state.llm_response = ''  # Reset the LLM response for the next interaction

# Display the conversation history
for speaker, text in st.session_state.history:
    st.text_area(label=speaker, value=text, height=75, disabled=True)

# Input for new questions
query_text = st.text_input('Enter your question:', key='query_text')

# Button to submit the question
submit_button = st.button('Submit', on_click=handle_chat)

if uploaded_file is not None and query_text:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        tmpfile_path = tmpfile.name

    # Document loading and text processing
    loader = PyPDFLoader(tmpfile_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Embeddings and vector database setup
    embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    retriever = vectordb.as_retriever()
    vectordb.persist()

    # LLM setup and query processing
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=512, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    st.session_state.llm_response = qa_chain.invoke(query_text)

    os.remove(tmpfile_path)