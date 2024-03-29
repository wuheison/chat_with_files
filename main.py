import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

print(f'cuda enabled?{torch.cuda.is_available()}')
# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# question template
template = """Question: {question}

Answer: Let's think step by step."""

question = PromptTemplate(template=template, input_variables=["question"])


# Streamlit UI setup
st.set_page_config(page_title='Chat with Files')
st.title('Chat with Files')

# Initialize session state variables if they don't exist
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File uploader for PDFs
uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

# Process the uploaded PDF file
if uploaded_file is not None and not st.session_state.pdf_processed:
    with st.spinner('Processing the PDF file...'):
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

        # Setup qa_chain with the updated retriever
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=64, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        vectordb.persist()

        # Cleanup temporary file
        os.unlink(tmpfile_path)
        st.session_state.pdf_processed = True
        st.success('PDF file processed and stored successfully!')


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if question := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            try:
                result = st.session_state.qa_chain(question)
                if "result" in result:
                    response = result["result"]
                else:
                    response = "Sorry, I couldn't process that question."
            except Exception as e:
                response = "An error occurred while processing your question."
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
