import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, ImageCaptionLoader, TextLoader, CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import bs4

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

# import chromadb

# chromadb.api.client.SharedSystemClient.clear_system_cache()

class ChatbotRAGFromText:
    def __init__(self, chat_model):
        self.store = {}
        self.chat_model = chat_model

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set in environment variables.")

        print("Loading Chat Model...")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=self.chat_model)
        print(f"Model {self.chat_model} loaded successfully!")

        print("Loading Embeddings...")
        try:
            os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("Embeddings loaded successfully!")
        except Exception as e:
            print("Error loading embeddings:", e)

        # print(self.embeddings)

    
    # def update_data_path(self, new_path):
    #     self.path = new_path
    #     self.define_ragchain(path=self.path)  # Reinitialize the chain with the new path


    def data_loader(self, path):
        
        tag = path.split(".")[-1]
        tag = tag.lower()
        if tag == 'png' or tag == 'jpg' or tag == 'jpeg':
            loader = ImageCaptionLoader(path)
        elif tag == 'pdf':
            loader = PyPDFLoader(path)
        elif tag == 'txt':
            loader = TextLoader(path)
        elif tag == 'csv':
            loader = CSVLoader(path)
        else:
            raise ValueError(f"Unsupported file format: {tag}")
        
        # YoutubeAudioLoader()

        # loader = TextLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        vector_store = FAISS.from_documents(documents=splits, embedding=self.embeddings)

        retriever = vector_store.as_retriever()
        return retriever
    

    def define_ragchain(self, path):
        ## Prompt Template
        system_prompt = (
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history "
            "formulate a stand alone question which can be understood "
            "without the chat history. Do not answer the question, "
            "just formulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        retriever = self.data_loader(path)

        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    def get_answer(self, message, session_id):
        def get_session_history(session_id) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        conversation_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        reply = conversation_rag_chain.invoke(
            {"input": f"{message}"},
            config= {
                "configurable": {"session_id": session_id}
            },
        )["answer"]

        return reply
