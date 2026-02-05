import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Document RAGbot", page_icon="📄")

def main():
    st.title("📄 Document RAGbot")
    st.markdown("### Phase 3: Semantic Indexing")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        # Save temp file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and Chunk (Phase 2)
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(pages)

        st.info(f"PDF split into {len(chunks)} chunks.")

        # --- PHASE 3: VECTOR STORAGE ---
        # We initialize the embedding model (this turns text into numbers)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if st.button("Index Document"):
            with st.spinner("Creating Vector Embeddings... this may take a minute."):
                # This creates the database and saves it locally in 'chroma_db'
                vectorstore = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                st.success("Success! Document indexed in ChromaDB.")
                st.balloons() # Just for fun!

        os.remove("temp.pdf")

if __name__ == "__main__":
    main()