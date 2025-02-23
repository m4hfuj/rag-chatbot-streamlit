import streamlit as st
import os
import tempfile
from chatbot_rag import ChatbotRAGFromText

# Streamlit UI
st.set_page_config(page_title="AI Chatbot")
st.title("📚 Multimodal RAG Chat")
# 🤖 About This Chatbot  
# #### 🔹 Key Features:  
# - **Multi-Model Support**: Choose from models like LLaMA 3, Gemma 2, and Mixtral for optimal performance.  
# - **Advanced Text Retrieval**: Uses embedding models to understand and fetch relevant information.  
# - **Document-Based Q&A**: Upload a file, and the chatbot will extract insights to answer your queries.  
# - **Chat History Memory**: Keeps track of past interactions for a seamless conversation flow.  

# Upload a file and start exploring AI-powered document analysis! 🚀

st.markdown("""
This AI-powered chatbot utilizes Retrieval-Augmented Generation (RAG) to provide intelligent responses based on uploaded documents. Simply upload a file, select your preferred models, and start chatting!
Source Code: https://github.com/m4hfuj/rag-chatbot-streamlit
""")


# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# CHAT_MODEL = "llama3-70b-8192"

# Sidebar for model selection
st.sidebar.title("Settings")
chat_model_options = ["llama3-70b-8192", "gemma2-9b-it", "qwen-2.5-32b", "mixtral-8x7b-32768", "deepseek-r1-distill-llama-70b"]  # Add more models if needed
CHAT_MODEL = st.sidebar.selectbox("Choose Chat Model", chat_model_options)

embeddings_options = ["all-MiniLM-L6-v2", "Alibaba-NLP/gte-Qwen2-7B-instruct"]
EMBEDDING_MODEL = st.sidebar.selectbox("Choose Embedding Model", embeddings_options)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

groq_api_key = st.secrets["GROQ_API_KEY"]
hf_token = st.secrets["HF_TOKEN"]
chatbot = ChatbotRAGFromText(CHAT_MODEL, EMBEDDING_MODEL, groq_api_key, hf_token)

# File Upload
uploaded_file = st.file_uploader("Upload a file")

if uploaded_file:
    # tag = uploaded_file.name.split(".")[-1]
    # name = f"temp.{tag}"
    # with open(name, "wb") as f:
    #     f.write(uploaded_file.read())

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filename = temp_file.name  # Store the temporary file path

    # print(uploaded_file.name)
    st.session_state.temp_filename = temp_filename
    chatbot.define_ragchain(temp_filename)
    st.session_state.chatbot = chatbot
    st.success("File loaded successfully! Start chatting below.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Type your question...")
# st.write("Made by m4hfuj")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    if "chatbot" in st.session_state:
        response = st.session_state.chatbot.get_answer(user_input, session_id="1234")
    else:
        response = "Please upload a PDF first."

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        if "deepseek" in CHAT_MODEL:
            thinking = response.split("</think>")[0]
            answer = response.split("</think>")[1]
            st.markdown(f'<div style="color: blue;">{thinking}</div>', unsafe_allow_html=True)
            st.write(answer)
        else:
            st.write(response)


# Handle cleanup when the app reloads
if "temp_filename" in st.session_state:
    import os
    if os.path.exists(st.session_state.temp_filename):
        os.remove(st.session_state.temp_filename)
        del st.session_state.temp_filename

