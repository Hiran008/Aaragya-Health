# drug_vectorstore.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DRUG_PATH = "data/Handbook of clinical drug data.pdf"
DRUG_DB_PATH = "vectorstore/db_drug_faiss"

loader = PyPDFLoader(DRUG_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DRUG_DB_PATH)
