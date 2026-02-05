# 🤖 Document RAGbot (Groq-Powered)

A high-performance Retrieval-Augmented Generation (RAG) application that allows users to chat with their PDF documents in real-time. Built with **Python**, **LangChain**, and **Groq**.

## 🚀 Features
- **Lightning-Fast Inference**: Utilizes Groq's LPU inference engine with Llama-3 for near-instant responses.
- **Semantic Search**: Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) to understand document context beyond simple keywords.
- **Local Vector Storage**: Implements ChromaDB to store and manage document embeddings locally.
- **User-Friendly UI**: Streamlit-based interface for easy document ingestion and conversational Q&A.

## 🏗️ Architecture
The app follows a standard RAG pipeline:
1. **Ingestion**: PDF is loaded and split into overlapping chunks (1000 characters).
2. **Embedding**: Chunks are converted into vectors using a sentence-transformer model.
3. **Storage**: Vectors are stored in a local ChromaDB instance.
4. **Retrieval**: When a user asks a question, the most relevant chunks are retrieved via similarity search.
5. **Generation**: The retrieved context + the user question are sent to Llama-3 (via Groq) to generate an accurate answer.



## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/CarterChamby/Document-RAGbot.git](https://github.com/CarterChamby/Document-RAGbot.git)
   cd Document-RAGbot
