import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

NIH_API = "https://ods.od.nih.gov/api/"
CHROMA_DB_PATH = "chroma_db/"

# Scrape links
response = requests.get(NIH_API)
soup = BeautifulSoup(response.text, "html.parser")
links = soup.find_all("a", string="HTML")
web_paths = [NIH_API + link["href"] for link in links if "español" not in link["href"]]

# Load docs
documents = []
for url in web_paths:
    loader = WebBaseLoader(web_paths=[url])
    loaded_docs = loader.load()
    for doc in loaded_docs:
        doc.metadata["source"] = url  # Attach source before splitting
        documents.append(doc)

# Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(documents)

# Re-attach source metadata to splits (sometimes lost)
for split in splits:
    if "source" not in split.metadata:
        split.metadata["source"] = split.metadata.get("url", "Unknown")

# Embed & store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=CHROMA_DB_PATH,
    client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
)
vectorstore.persist()
