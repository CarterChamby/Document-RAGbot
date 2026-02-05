import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# NEW: Import for Groq
from langchain_groq import ChatGroq 
from langchain.chains import RetrievalQA

load_dotenv()

st.set_page_config(page_title="Groq RAGbot", page_icon="⚡")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def main():
    st.title("⚡ Groq-Powered RAGbot")
    embeddings = get_embeddings()

    # --- SIDEBAR: Document Management ---
    with st.sidebar:
        st.header("1. Data Ingestion")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file and st.button("Index Document"):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(pages)
            
            # Create/Load the Vector Store
            vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            st.success("Indexed with ChromaDB!")
            os.remove("temp.pdf")

    # --- MAIN AREA: Chat Interface ---
    st.header("2. Instant Q&A")
    
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        # NEW: Initialize Groq LLM (Llama-3)
        llm = ChatGroq(
            model_name="llama3-8b-8192", 
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        user_query = st.text_input("Ask a question:")
        
        if user_query:
            with st.spinner("Groq is processing..."):
                response = qa_chain.invoke(user_query)
                st.markdown("### Answer:")
                st.write(response["result"])
    else:
        st.info("Upload a document in the sidebar to get started!")

if __name__ == "__main__":
    main()