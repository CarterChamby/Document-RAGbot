import streamlit as st
import os
from dotenv import load_dotenv

# Document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store & embeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LLM
from langchain_groq import ChatGroq

# LCEL / Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------
# App setup
# -------------------------------------------------
load_dotenv()
st.set_page_config(page_title="Document RAGbot", page_icon="🤖")

# -------------------------------------------------
# Cache models
# -------------------------------------------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    return embeddings, llm


# -------------------------------------------------
# Main app
# -------------------------------------------------
def main():
    st.title("🤖 Document RAGbot (Groq Engine)")
    embeddings_model, llm = load_models()

    # ---------------- Sidebar: Upload & index ----------------
    with st.sidebar:
        st.header("1. Document Setup")
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded_file and st.button("Index Document"):
            with st.spinner("Processing document..."):
                # Save temporary file
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load & split
                loader = PyPDFLoader("temp.pdf")
                pages = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                )
                chunks = splitter.split_documents(pages)

                # Create / persist vector DB
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings_model,
                    persist_directory="./chroma_db",
                )

                os.remove("temp.pdf")
                st.success("Indexing complete!")

    # ---------------- Main: Chat ----------------
    st.header("2. Chat with your Data")

    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings_model,
        )

        retriever = vectorstore.as_retriever()

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # ---------------- LCEL RAG chain ----------------
        rag_chain = (
            {
                "context": retriever,
                "input": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        user_input = st.text_input("Ask a question about the PDF:")

        if user_input:
            with st.spinner("AI is thinking..."):
                answer = rag_chain.invoke(user_input)
                st.markdown("### Answer:")
                st.write(answer)

    else:
        st.info("Upload and index a PDF to begin.")


if __name__ == "__main__":
    main()
