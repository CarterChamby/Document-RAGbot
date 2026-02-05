import streamlit as st
import os
from dotenv import load_dotenv

# Document Processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store & Embeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LLM & RAG Chain
# UPDATE: Fixed import path for modern LangChain
from langchain_groq import ChatGroq 
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load API Keys
load_dotenv()

st.set_page_config(page_title="Document RAGbot", page_icon="🤖")

# --- PRO-TIP: Cache the Models ---
@st.cache_resource
def load_models():
    # Load free local embeddings (sentence-transformers)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize Groq LPU (High speed)
    llm = ChatGroq(
        model_name="llama3-8b-8192", 
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    return embeddings, llm

def main():
    st.title("🤖 Document RAGbot (Groq Engine)")
    embeddings_model, llm = load_models()

    # --- SIDEBAR: Upload & Index ---
    with st.sidebar:
        st.header("1. Document Setup")
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        
        if uploaded_file:
            if st.button("Index Document"):
                with st.spinner("Processing..."):
                    # Save temp file for loader
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Phase 2: Loading & Chunking
                    loader = PyPDFLoader("temp.pdf")
                    pages = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = splitter.split_documents(pages)
                    
                    # Phase 3: Vector Storage
                    # This saves to your 'chroma_db' folder
                    vectorstore = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embeddings_model,
                        persist_directory="./chroma_db"
                    )
                    st.success("Indexing Complete!")
                    os.remove("temp.pdf")

    # --- MAIN AREA: Chat Interface ---
    st.header("2. Chat with your Data")
    
    # Check if a database exists before allowing chat
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
        
        # Define the Prompt (The instructions for the AI)
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Setup the modern RAG Chain
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

        user_input = st.text_