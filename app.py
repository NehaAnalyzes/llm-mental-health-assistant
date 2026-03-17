import os
import hashlib
import json
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

DATA_DIR = "data/"
CHROMA_DIR = "chroma_db"
HASH_FILE = "data_hash.json"

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

def get_data_hash():
    hasher = hashlib.md5()
    pdf_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])
    for fname in pdf_files:
        with open(os.path.join(DATA_DIR, fname), "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()

@st.cache_resource
def load_vectordb():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    current_hash = get_data_hash()

    if os.path.exists(CHROMA_DIR) and os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            saved = json.load(f)
        if saved.get("hash") == current_hash:
            return Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embedding_model
            )

    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=CHROMA_DIR
    )

    with open(HASH_FILE, "w") as f:
        json.dump({"hash": current_hash}, f)

    return vectordb

vectordb = load_vectordb()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.7
)

st.title("🌿 Valsu — Mental Health Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    docs = retriever.invoke(prompt)
    context = "\n\n".join([d.page_content for d in docs])

    full_prompt = f"""
You are a warm mental health assistant.

Context:
{context}

User: {prompt}
Assistant:
"""

    response = llm.invoke(full_prompt).content

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)