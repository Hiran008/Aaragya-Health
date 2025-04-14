import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS, Chroma
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Paths
DB_FAISS_PATH = "vectorstore/db_faiss"
CHROMA_DB_PATH = "chroma_db/"
DRUG_DB_PATH = "vectorstore/db_drug_faiss"

st.set_page_config(page_title="Aragya-Health", layout="wide")

# ---- HEADER (Always Static) ----
st.markdown("<h1 style='text-align: center;'>🧠 Welcome to Aaragya-Health</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---- Session State Init ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_vectorstores():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss = FAISS.load_local(DB_FAISS_PATH, embedding, allow_dangerous_deserialization=True)
    chroma = Chroma(
        embedding_function=embedding,
        persist_directory=CHROMA_DB_PATH,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
    drugs = FAISS.load_local(DRUG_DB_PATH, embedding, allow_dangerous_deserialization=True)
    return faiss.as_retriever(search_kwargs={'k': 3}), chroma.as_retriever(search_kwargs={'k': 3}), drugs.as_retriever(search_kwargs={'k': 3})

@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"max_length": "512"},
        huggingfacehub_api_token=HF_TOKEN
    )

def build_rag_chain(retriever, llm, source=""):
    prompt_template = """
    You are a helpful medical assistant. Use the context below to answer the question.
    If you don't know the answer, say you don't know. Do NOT hallucinate.
    Limit your response to 3 sentences max.

    Source: {source}
    Context: {context}
    Question: {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "source"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt.partial(source=source)},
    )

def build_drug_chain(retriever, llm):
    drug_prompt_template = """
    You are a helpful and cautious medical assistant. Your task is to recommend drugs based ONLY on the provided context from a verified drug handbook.

    If the context does not contain any suitable drug information related to the user's question, say clearly: "I'm sorry, I couldn't find a specific drug recommendation based on the information available."

    Never make up drug names or suggestions. Use only the drugs and data found in the provided context.

    Context: {context}
    Question: {question}
    """
    prompt = PromptTemplate(template=drug_prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

# Load
faiss_retriever, chroma_retriever, drug_retriever = load_vectorstores()
llm = load_llm()
faiss_chain = build_rag_chain(faiss_retriever, llm, source="PDF")
chroma_chain = build_rag_chain(chroma_retriever, llm, source="NIH")
drug_chain = build_drug_chain(drug_retriever, llm)

# Chat Input
user_input = st.chat_input("Welcome, ask your medical query...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # RAG Responses
    response_pdf = faiss_chain.invoke({"query": user_input})
    response_nih = chroma_chain.invoke({"query": user_input})
    response_drug = drug_chain.invoke({"query": user_input})

    combined_answer = (
        f"**📄 From Medical-Encyclopedia:** {response_pdf['result']}\n\n"
        f"**🌐 From NIH:** {response_nih['result']}\n\n"
        f"**💊 Drug Recommendation:** {response_drug['result']}"
    )

    sources = (
        [doc.metadata for doc in response_pdf["source_documents"]]
        + [doc.metadata for doc in response_nih["source_documents"]]
        + [doc.metadata for doc in response_drug["source_documents"]]
    )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": combined_answer,
        "sources_title": "📄 PDF + 🌐 NIH + 💊 Drug Sources",
        "sources": sources
    })

# Display Chat History after user input to keep source dropdowns
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander(msg["sources_title"]):
                for src in msg["sources"]:
                    st.markdown(f"- [{src.get('source', 'Link')}]({src.get('source', '#')})")