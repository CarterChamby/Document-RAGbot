import streamlit as st
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Set up the Page Config (This makes the browser tab look nice)
st.set_page_config(page_title="DocuMind AI", page_icon="📄")

def main():
    st.title("📄 DocuMind: Your Research Assistant")
    st.subheader("Upload PDFs and ask questions")

    # Sidebar for file uploads
    with st.sidebar:
        st.header("Upload Center")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if uploaded_files:
                st.success("Files received! Ready for Phase 2.")
            else:
                st.error("Please upload at least one PDF.")

    # Chat interface placeholder
    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.chat_message("user").write(query)
        st.chat_message("assistant").write("I'm ready to learn! Let's implement Phase 2 to get me thinking.")

if __name__ == "__main__":
    main()