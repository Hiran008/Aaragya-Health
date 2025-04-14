# 🧠 Aaragya-Health: A RAG-based Healthcare Chatbot

Aaragya-Health is an intelligent medical assistant that leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses to health-related queries. By integrating trusted medical documents and advanced language models, Aaragya-Health aims to assist users with reliable information and drug recommendations.

---

## 🚀 Features

- **Conversational AI**: Engages users in natural language conversations to address their medical queries.
- **Retrieval-Augmented Generation (RAG)**: Combines retrieval mechanisms with generative models to provide context-rich answers.
- **Multi-Source Integration**: Utilizes data from PDFs, NIH articles, and drug handbooks for comprehensive responses.
- **Streamlit Interface**: Offers an intuitive and user-friendly web interface for seamless interactions.
- **Local Deployment**: Ensures data privacy by running entirely on local machines without external dependencies.

---

## 🧰 Technologies Used

- **[LangChain](https://github.com/langchain-ai/langchain)**: Framework for developing applications powered by language models.
- **[Hugging Face Transformers](https://huggingface.co/transformers/)**: Provides pre-trained models like Mistral-7B-Instruct-v0.3 for text generation.
- **[FAISS](https://github.com/facebookresearch/faiss)**: Efficient similarity search and clustering of dense vectors.
- **[ChromaDB](https://www.trychroma.com/)**: Embedding database for storing and querying vector representations.
- **[Streamlit](https://streamlit.io/)**: Framework for building interactive web applications in Python.
- **[Python-dotenv](https://github.com/theskumar/python-dotenv)**: Reads key-value pairs from a `.env` file and can set them as environment variables.

---

## 🏠 Architecture

```plaintext
User Query
   │
   ▼
Streamlit Interface
   │
   ▼
Retrieval Mechanism (FAISS & ChromaDB)
   │
   ▼
Relevant Context Extraction
   │
   ▼
Hugging Face LLM (Mistral-7B-Instruct-v0.3)
   │
   ▼
Generated Response
   │
   ▼
Display to User
```

1. **User Interaction**: Users input their medical queries through the Streamlit web interface.
2. **Context Retrieval**: The system retrieves relevant information from embedded medical documents using FAISS and ChromaDB.
3. **Response Generation**: The retrieved context is passed to a Hugging Face language model to generate a coherent and informative response.
4. **Display**: The generated answer is displayed back to the user through the Streamlit interface.

---

## 📄 Data Sources

- **Handbook of Clinical Drug Data**: Provides comprehensive drug information.
- **NIH Articles**: Trusted medical articles from the National Institutes of Health.
- **Medical PDFs**: Additional medical documents for enriched context.

---

## 🛠️ Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/aaragya-health.git
   cd aaragya-health
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add your Hugging Face API token:
     ```
     HF_TOKEN=your_huggingface_api_token
     ```

5. **Prepare the Data**:
   - Place your medical PDFs in the `data/` directory.
   - Run the script to process and embed the documents:
     ```bash
     python drug_vectorstore.py
     ```

6. **Run the Application**:
   ```bash
   streamlit run medibot.py
   ```

---

## 🗀️ Screenshots

![Aaragya-Health Interface](screenshots/interface.png)
*Figure: Aaragya-Health Streamlit Interface*

---

## 📬 Contact

For any inquiries or feedback, please contact [hiranmoy88dey@gmail.com](mailto:hiranmoy88dey@gmail.com).


