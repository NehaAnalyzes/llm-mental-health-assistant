import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

st.set_page_config(page_title="Mental Health Assistant", page_icon="🌿")

# =========================
# LOAD VECTOR DB (FAST CACHE)
# =========================
@st.cache_resource(show_spinner=True)
def load_vectordb():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = []
    data_path = "data"

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,   # smaller = faster embedding
        chunk_overlap=40
    )

    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        embedding_model
    )

    return vectordb


vectordb = load_vectordb()
retriever = vectordb.as_retriever(search_kwargs={"k": 2})  # reduced for speed

# =========================
# LOAD GEMINI (SAFE MODEL)
# =========================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0.6
)

# =========================
# UI
# =========================
st.title("🌿 Mental Health Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    docs = retriever.invoke(prompt)
    context = "\n\n".join([d.page_content[:800] for d in docs])  # limit context size

    full_prompt = f"""
You are a warm, empathetic mental health assistant.
Use the context to provide supportive responses.

Context:
{context}

User: {prompt}
Assistant:
"""

    try:
        response = llm.invoke(full_prompt).content
    except Exception as e:
        response = f"LLM ERROR: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)