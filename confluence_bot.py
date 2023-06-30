# confluence_bot.py

## Python Setup
import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

## Environment Setup
# Exception Handling
try:
    # Load up the bot environment
    load_dotenv(override=True)
    # Obtain API key from .env file
    OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
except:
    raise KeyError(
        "ERROR: Unable to find OpenAI API key in .env file. Check key is assigned to OPEN_API_KEY"
    )


## Model Setup
@st.cache_resource
def setup_model():
    # Create the embeddings and doc search
    folder_path = "data/docs"
    txt_loader = DirectoryLoader(
        folder_path, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader
    )
    txt_documents = txt_loader.load()
    text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
    texts = text_splitter.split_documents(txt_documents)
    # Model objects for chatbot querying
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Chroma.from_documents(texts, embeddings)

    # Return the docsearch object
    return docsearch


## Update model function
def update_model(docsearch_obj, model_temperature: int = 0.0):
    # Create the LLM with the temperature selection
    llm = OpenAI(
        model_name="text-davinci-003",
        temperature=model_temperature,
        openai_api_key=OPENAI_API_KEY,
    )
    qa_chain = VectorDBQA.from_chain_type(
        llm=llm, chain_type="stuff", vectorstore=docsearch_obj
    )

    # Return
    return qa_chain


## Streamlit App Section
# Create the intial docsearch object
docsearch_obj = setup_model()

# Define Streamlit app layout
st.title("Confluence Q&A Bot")

# Temperature Control
# Header
st.header("Model Parameters")
# Temperature variable control (slider)
temperature_values = st.slider(
    "Select a temperature for the model",
    min_value=0.0,
    max_value=1.0,
    step=0.1,
    value=0.3,
)
# Information on temperature variable (expanded)
temperature_info = st.expander("What is the temperature parameter?")
temperature_info.write(
    """The LLM temperature is a hyperparameter that regulates the randomness, or creativity, of the AI's responses.
    A higher temperature value typically makes the output more diverse and creative but might also increase its likelihood of straying from the context.
    """
)

# Chatbot Window
# Header
st.header("Documentation ChatBot")

# Initialise the model
qa_chain = update_model(docsearch_obj, temperature_values)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user unput
if q_input := st.chat_input(
    "What would you like to know about the CVD Prevent Tool codebase?"
):
    # User Prompt
    # Display user message in chat message container
    st.chat_message("User").markdown(q_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "User", "content": q_input})

    # Display assistant response in chat message container
    with st.chat_message("Confluence Bot"):
        # Placeholder Response
        message_placeholder = st.empty()
        # Confluence Bot Response
        assistant_response = qa_chain.run(f"{q_input}")
        full_response = ""
        # Simulate stream of response
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "Confluence Bot", "content": full_response}
    )
