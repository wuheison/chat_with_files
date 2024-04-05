import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import boto3
from botocore.exceptions import ClientError
import json

# for aws secrete manager on ec2
def get_aws_secret():
    secret_name = "API_tokens"
    region_name = "ap-southeast-1"

    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
    
    secret_string = get_secret_value_response['SecretString']
    secret_dict = json.loads(secret_string)
    api_key = secret_dict['HUGGINGFACEHUB_API_TOKEN']
    print(f'api key is {api_key}')
    return api_key


HUGGINGFACEHUB_API_TOKEN = get_aws_secret()


# question template
template = """Question: {question}

Answer: Let's think step by step. 
Find the answer only from the retriever.
If you cannot find or don't know the answer, do not guess, just say you cannot find the answer."""

question = PromptTemplate(template=template, input_variables=["question"])


# Streamlit UI setup
st.set_page_config(page_title='AI Chat with your Files')
st.title('AI Chat with your Files')

# Description for users with Markdown for styling
st.markdown("""
## Use AI to Chat with your **PDF**, **Word**
""")

# Using columns for layout
col1, col2 = st.columns(2)

with col1:
    st.image("asset/logo.jpg")

with col2:
    st.markdown("""
    ### Instructions:
    1. **Drag and drop** or Click on the **Browse files** button below.
    2. **Wait** for processing.
    3. Type your **question** in the chat input.
    4. Get **answers** based on your file's content.
    5. **Refresh** to restart
    """)
# Initialize session state variables if they don't exist
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File uploader for PDFs
uploaded_file = st.file_uploader("Upload a PDF or Word file", type=['pdf','docx'])

# Process the uploaded file
if uploaded_file is not None and not st.session_state.pdf_processed:
        
        with st.spinner('Processing file...'):
            file_type = uploaded_file.name.split('.')[-1]  # Get the file extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                tmpfile_path = tmpfile.name

            # Choose the appropriate loader based on the file type
            if file_type == 'pdf':
                loader = PyPDFLoader(tmpfile_path)
            elif file_type == 'docx':
                loader = Docx2txtLoader(tmpfile_path)
            else:
                st.error("Unsupported file type")
                st.stop()

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
            st.success('File processed and stored successfully!')


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
