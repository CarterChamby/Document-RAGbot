import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Document RAGbot", page_icon="📄")

def main():
    st.title("PDF Ingestion & Chunking")
    st.write("This prepares the document for the AI's 'memory'.")

    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        # We must save the uploaded file to a temporary local file 
        # so the PyPDFLoader can access its path.
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. Loading the PDF
        # This converts the PDF format into a list of LangChain Document objects
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()

        # 3. Splitting/Chunking
        # 'chunk_size' is how many characters are in each piece.
        # 'chunk_overlap' ensures the AI doesn't lose context between pieces.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_documents(pages)

        # 4. Display Results
        st.success(f"Successfully split the PDF into {len(chunks)} chunks!")
        
        # Let's peek at what the AI sees
        with st.expander("See Example Chunk"):
            st.write(f"**Content of Chunk 1:**")
            st.write(chunks[0].page_content)
            st.write(f"**Metadata:** {chunks[0].metadata}")

        # Clean up: delete the temp file so it doesn't clutter your project
        os.remove("temp.pdf")

if __name__ == "__main__":
    main()