from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("data", file))
        documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=40
)

chunks = splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory="chroma_db"
)

print("Vector DB created successfully.")