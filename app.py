import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader

import os

# Function to load documents 
def load_documents():
    loader = DirectoryLoader('data/', glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

# Function to split text into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to create embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    return embeddings

# Function to create vector store
def create_vector_store(text_chunks, embeddings):
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

# Function to create LLMS model
def create_llms_model():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    llm = HuggingFaceHub(
        repo_id="somosnlp/Phi-2-LenguajeClaro",
        model_kwargs={
        "max_new_tokens": 250,  # Ajusta según sea necesario
        # O bien, si prefieres aumentar max_length
        # "max_length": 512,
        "repetition_penalty": 1.1,
        "temperature": 0.5,
        "top_p": 0.9,
        "return_full_text": False
    }
        # model_kwargs={
        #     "max_new_tokens": 512,
        #     "repetition_penalty": 1.1,
        #     "temperature": 0.5,
        #     "top_p": 0.9,
        #     "return_full_text": False
        # }
    )
    return llm
st.set_page_config(
    page_title="IEEE CIS UNI ChatBot",
    page_icon="./logoieeecisuni.jpg"
)
# Initialize Streamlit app
st.markdown(
    """
    <style>
    .title {
        color: #01285D;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>👾 IEEE CIS UNI ChatBot 🤖</h1>", unsafe_allow_html=True)
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.markdown('<style>h3{color: pink; text-align: center;}</style>', unsafe_allow_html=True)

# loading of documents
documents = load_documents()

# Split text into chunks
text_chunks = split_text_into_chunks(documents)

# Create embeddings
embeddings = create_embeddings()

# Create vector store
vector_store = create_vector_store(text_chunks, embeddings)

# Create LLMS model
llm = create_llms_model()

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

# Define chat function
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hola hasme una pregunta sobre IEEE CIS UNI 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hola!"]

# Display chat history
reply_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Question:", placeholder="Has tu pregunta sobre IEEE CIS UNI", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversation_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        
with st.sidebar:
    st.image("logoieeecisuni.jpg", use_column_width=True)
    st.write("Este chatbot es un proyecto en constante mejora de IEEE CIS UNI, nos permite saber más sobre IEEE CIS UNI atraves de este chat")
    
if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="pixel-art")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
