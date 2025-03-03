import streamlit as st
import os
from gtts import gTTS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
#from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file
load_dotenv()

# Initialize Groq for chat pdf
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.sidebar.error("GROQ_API_KEY is not set. Please set it in the .env file.")
    st.stop()

model = 'llama-3.1-70b-versatile'

groq_chat = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name=model
)


embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# Define functions for file handling and processing
def save_uploaded_files(uploaded_files, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for uploaded_file in uploaded_files:
        with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    return save_dir

def pdf_read(pdf_directory):
    loader = PyPDFDirectoryLoader(pdf_directory)
    data = loader.load()
    return data

# Return vectorstore for the documents
def get_vector_store(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = text_splitter.split_documents(data)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    return "output.mp3"



# Returns history_retriever_chain
def get_retriever_chain(vector_store):
    llm = groq_chat
    retriever = vector_store.as_retriever(search_kwargs={'k': 20})
    #compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=rerank_model, top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", """Based on the above conversation, generate a search query that retrieves the most relevant and up-to-date information for the user. Focus on key topics, entities, or concepts that are directly related to the user's query. 
        Make sure the search query is specific and targets the most relevant sources of information.""")
    ])
    history_retriever_chain = create_history_aware_retriever(llm, compression_retriever, prompt)

    return history_retriever_chain

# Returns conversational rag
def get_conversational_rag(history_retriever_chain):
    llm = groq_chat
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a highly knowledgeable assistant, your task is to answer any task or query of the user, using information retrieved from provided PDF documents.
        Your goal is to provide clear and accurate answers based on the retrieved context. 
        If the answer is not directly available, say: "I couldn't find this information in the provided documents."
        Be concise, but thorough.
        \n\nContext snippets used in response:\n\n{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, answer_prompt)

    # Create final retrieval chain
    conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)

    return conversational_retrieval_chain

# Returns the final response
def get_response(user_input):
    history_retriever_chain = get_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag(history_retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response["answer"]


# Main app
def main():
    st.set_page_config("AI Assistant📝")
    st.header("AI Assistant📝")

    # Sidebar
    with st.sidebar:
        st.title("Menu:")
        pdf_files = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        save_dir = "uploaded_pdfs"

    if pdf_files:
        save_uploaded_files(pdf_files, save_dir)
        st.sidebar.success("PDF Uploaded and Processed")

        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        raw_text = pdf_read(save_dir)

        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vector_store(raw_text)
            st.session_state.chat_history = [AIMessage(content="Hi, how can I help you?")]       
            
            # User input through chat interface
        user_input = st.chat_input("Type your message here...")
        if user_input is not None and user_input.strip() != "":
            response = get_response(user_input)
        
                # Update chat history
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response))
            

            # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
                    audio_file = text_to_speech(message.content)
                    st.audio(audio_file, format="audio/mp3")
                        
            else:
                with st.chat_message("Human"):
                    st.write(message.content)
           


if __name__ == "__main__":
    main()