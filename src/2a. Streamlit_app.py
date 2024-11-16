# https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context/
# https://docs.llamaindex.ai/en/stable/examples/cookbooks/oreilly_course_cookbooks/Module-2/Components_Of_LlamaIndex/

import warnings
warnings.filterwarnings("ignore")

import re
import streamlit as st
from streamlit import session_state
import time
from datetime import datetime
import base64
import re
import streamlit as st
import time
import pandas as pd
import chromadb
import torch
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


model_path = 'microsoft/Phi-3.5-mini-instruct'

DB_DIR = "./chroma_db"
DB_NAME = 'DB_01'

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#Reqd functions
def get_chat_eng(index, llm, chat_mode='condense_plus_context'):
    # print("went inside get_chat_eng")
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900) #token_limit=3900
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about data related to first two chapters of Biology subject."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),

    chat_engine = index.as_query_engine(
        memory=memory,
        llm=llm,
        context_prompt=context_prompt,
        verbose=False,
        streaming=False,
        similarity_top_k = 5
    )
    return chat_engine


@st.cache_resource(show_spinner=False)
def get_vectorstore_index():
    # print("index retrieved")
    db2 = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = db2.get_or_create_collection(DB_NAME)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    return index

@st.cache_resource(show_spinner=False)
def llm_model():    
    Settings.llm = HuggingFaceLLM(
        model_name=model_path,
        tokenizer_name=model_path,
        context_window=3900,
        max_new_tokens=2000,
        model_kwargs={"torch_dtype": torch.float16},
        generate_kwargs={"temperature": 0.05,  "top_k": 5, "top_p": 0.9},
        device_map='auto'
    )

    return Settings.llm

def get_source_df(llm_response, threshold = 0.45):
   source_df = pd.DataFrame()
   pg_no, file_name, file_path, score = [], [], [], []

   for node in llm_response.source_nodes:
      scr = node.score

      if scr >= threshold:
         f_name = node.metadata['file_name']
         p_no = node.metadata['page_label']
         f_path = node.metadata['file_path']
         file_name.append(f_name)
         pg_no.append(p_no)
         file_path.append(f_path)
         score.append(scr)

   src_dict = {
      'File Name' : file_name,
      'Page Number' : pg_no,
      'File Path' : file_path,
      'Confidence Score' : score
   }

   source_df = pd.DataFrame(src_dict)

   return source_df

def get_chat_eng_(index, llm, chat_mode='condense_plus_context'):
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900) #token_limit=3900
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about data related to HO Policy documents."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),

    chat_engine = index.as_chat_engine(
        chat_mode= chat_mode, #  condense_plus_context
        memory=memory,
        llm=llm,
        context_prompt=context_prompt,
        verbose=True,
        streaming=True,
        similarity_top_k = 5
    )
    return chat_engine

def response_generator_llm(assistant_response):

    for word in re.split(r'(\s+)', assistant_response):
        yield word + " "
        time.sleep(0.01)

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    # pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def change_chatbot_style():
    # Set style of chat input so that it shows up at the bottom of the column
    chat_input_style = f"""
    <style>
        .stChatInput {{
          position: fixed;
          bottom: 3rem;
        }}
    </style>
    """
    st.markdown(chat_input_style, unsafe_allow_html=True)


# Streamlit Chat Interface
container_ht = 600

# manage sidebar
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'
st.set_page_config(layout="wide", initial_sidebar_state=st.session_state.sidebar_state)

st.title("🔍 RAG ChatBot")

llm = llm_model()
index = get_vectorstore_index()
chat_engine = get_chat_eng(index, llm, chat_mode='condense_question')

# Initialize chat history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
    

# intialize login
if 'login' not in session_state:
    session_state.login = False


# Login UI
with st.sidebar:
 name = st.text_input('Name')
 username = st.text_input('User-ID')

 if st.button('Login'):
    st.session_state.conversation_history = []    
    st.header(f"Welcome {name} to this Demo!!!")

 if username:
    session_state.login = True
    session_state.name = name
    session_state.username = username
    st.session_state.sidebar_state = 'collapsed'
 else:
    st.warning('Please enter valid Name and User-ID')


# Check login state
if session_state.login == True:

    col1, col2 = st.columns(2, gap='small')

    with col1.container(height=container_ht, border=False):

        # Display chat messages from history on app rerun
        for message in st.session_state.conversation_history:            
                if message['role']=='user':
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                else:
                    with st.chat_message(message["role"]):
                        st.markdown('Refer below table for details of source information:')
                        st.dataframe(message['table'], hide_index=True)
                        st.markdown(message["content"])

        # React to user input
        change_chatbot_style() #keep chat input always at bottom
        if prompt := st.chat_input("Enter your query 💬"):

            # Display user message in chat message container
            st.session_state.conversation_history.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):

                ### INDEX AS CHAT ENGINE
                # print(f"prompt: {prompt, type(prompt)}")    
                llm_response = chat_engine.chat(prompt)                
                
                # print(f"llm_response: {llm_response.source_nodes}")
                ans = llm_response.response

                source_df = get_source_df(llm_response)
                source_df_for_display = source_df[['File Name', 'Page Number']]
                # print(f'source_df: \n{source_df}')

                if len(source_df) > 0:
                    # print("went inside if")
                    src_table_header = st.markdown('Refer below table for details of source information:')
                    src_table = st.dataframe(source_df_for_display, hide_index=True)

                    comb_response = '\n\n' + ans
                    response = st.write_stream(response_generator_llm(ans))

                    with col2.container(height=container_ht, border=False):
                        filename_list = list(source_df['File Name'].unique())
                        tab = st.tabs(filename_list)

                        for i in range(len(filename_list)):
                            with tab[i]:
                                st.header('Source document:')
                                filepath = source_df[source_df['File Name'] == filename_list[i]]['File Path'].values[0]
                                displayPDF(filepath)
                else:
                    # print("went inside else")
                    src_table_header, source_df_for_display = '', ''
                    comb_response = ans + f'Please either elaborate the question or check if the given context is present in HO Policies document'
                    response = st.write_stream(response_generator_llm(comb_response))

            # Add assistant response to chat history
            st.session_state.conversation_history.append({"role": "assistant", "content": response, 'table': source_df_for_display})
            
