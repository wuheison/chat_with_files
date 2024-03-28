import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# question template
template = """Question: {question}

Answer: Let's think step by step."""

question = PromptTemplate(template=template, input_variables=["question"])

# Huggingface LLM
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=64, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN)

llm_chain = LLMChain(prompt=question, 
                     llm=llm)


# Streamlit UI setup
st.set_page_config(page_title='Chat with Files')
st.title('Chat with Files')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if question := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            llm_response = llm_chain.run(question)
            response = llm_response  # Assuming llm_response is the text response you want to display

        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
