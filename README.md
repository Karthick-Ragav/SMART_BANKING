Multimodal RAG Chatbot

📌 Overview

This project is a **Multimodal Retrieval-Augmented Generation (RAG) Chatbot** designed to answer user queries based on domain-specific documents such as:

* Home Loan Products
* Fixed Deposit Products
* Credit Card & Personal Loan Products
* Regulatory Disclosures & Compliance

The system ingests documents, processes them into structured chunks, stores them in a vector database, and retrieves relevant context to generate accurate responses.

---

⚙️ Key Features

* 📄 Document ingestion pipeline
* 🔍 Semantic search using vector embeddings
* 🤖 Context-aware chatbot responses
* 📊 Support for structured and unstructured data
* 📦 Scalable architecture for enterprise knowledge bases

---

🏗️ Architecture Overview

1. Document Upload
2. Ingestion Pipeline
3. Vector Database Storage
4. Retriever + LLM
5. Response Generation

---

📥 Ingestion Pipeline

The ingestion pipeline is a core part of the system and includes:

1. Docling Parser

* Extracts structured content from documents (PDFs, etc.)
* Converts raw documents into machine-readable format
* Handles text, tables, and layout-aware parsing

2. Chunking

* Splits extracted content into smaller chunks
* Maintains optimal chunk size and overlap
* Ensures better retrieval accuracy

3. Embedding Generation

* Converts chunks into vector representations
* Enables semantic similarity search

---

🗄️ Database (Vector Store)

* Stores embeddings along with metadata
* Supports fast similarity search
* Used during query time to fetch relevant chunks

Stored Data Includes:

* Text chunks
* Document metadata
* Embedding vectors

---

📤 Uploading Module

* Allows users to upload documents into the system
* Triggers ingestion pipeline automatically
* Ensures documents are processed and stored efficiently

---

🔎 Retrieval & Response

1. User query is converted into embedding
2. Relevant chunks are retrieved from the vector database
3. Context is passed to the LLM
4. Final response is generated

---

🚀 Tech Stack

* Python
* LangChain / LangGraph
* Vector Database
* LLM (Gemini / OpenAI / Cohere)
* Docling Parser

---

🧠 Contribution

This project focuses on building a robust ingestion pipeline and integrating it with a scalable RAG system.

---

📄 License

This project is for learning and demonstration purposes.
