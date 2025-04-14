# 🧠 Aaragya-Health: A RAG-based Healthcare Chatbot

Aaragya-Health is an intelligent, domain-aware healthcare chatbot designed to provide informative and safe responses to medical queries using Retrieval-Augmented Generation (RAG). It leverages high-quality sources including medical PDFs, NIH (National Institutes of Health) web content (via API access), and a verified drug handbook to recommend condition-specific drugs. The chatbot ensures safe and evidence-based outputs by combining retrieval mechanisms with a powerful large language model (LLM).

---

## 🚀 Features
- 🔍 Retrieval from medical PDFs, NIH articles (via API), and a drug handbook.
- 💬 Conversational chatbot interface with persistent chat history.
- 💊 Intelligent drug recommendation system.
- 🫠 Uses HuggingFace's Mistral-7B-Instruct-v0.3 for text generation.
- 📚 Transparent source citation from retrieved documents.

---

## 🧰 Technologies Used

| Tool/Technology | Purpose |
|-----------------|---------|
| **Streamlit** | Web interface for the chatbot |
| **LangChain** | Orchestration of LLM + retrieval logic |
| **HuggingFace (Mistral-7B)** | Backbone LLM for generating answers |
| **FAISS** | Vector similarity search on embedded PDFs and drug data |
| **ChromaDB** | Vector store for NIH web-scraped content |
| **SentenceTransformers** | Embedding model (MiniLM) for creating vector representations |
| **dotenv** | Secure API key and environment variable handling |
| **NIH API Access** | Enables retrieval of real-time NIH content using LangChain's `WebBaseLoader` |

---

## ⚙️ Architecture

Aaragya-Health is built on the Retrieval-Augmented Generation (RAG) architecture:

1. **Input Query**: User types a health-related question.
2. **Retrieval Layer**:
   - **FAISS** retrieves context from embedded PDFs.
   - **ChromaDB** pulls info from NIH web pages (via LangChain + NIH API).
   - **FAISS** (second instance) fetches relevant drug data from the drug handbook.
3. **LLM Reasoning**: Mistral-7B generates a concise response using retrieved context.
4. **Response Aggregation**: The system presents responses from all sources and their citations.
5. **Output**: Displayed via Streamlit with interactive source references.

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
https://github.com/Hiran008/Aaragya-Health.git
cd Aaragya-Health
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory and add:
```env
HF_TOKEN=your_huggingface_api_key
```

### 4. Prepare Vector Stores
Use the following scripts:
```bash
# Build vectorstore from medical PDFs
python create_pdf_vectorstore.py

# Build vectorstore from drug handbook
python drug_vectorstore.py

# Scrape and embed NIH content
python create_nih_vectorstore.py
```

### 5. Run the Application
```bash
streamlit run medibot.py
```

---

## 👌 Acknowledgments
- NIH (https://www.nih.gov/)
- HuggingFace & Mistral AI
- LangChain team
- SentenceTransformers by UKPLab

---

Enjoy using Aaragya-Health – your AI-powered companion for reliable healthcare information! 💺️🤖

