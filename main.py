import os
import hashlib
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import gradio as gr

# --- 1. Load environment variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

DATA_DIR = "data/"
CHROMA_DIR = "chroma_db"
HASH_FILE = "data_hash.json"


# --- 2. Smart DB loading (only rebuilds if PDFs changed) ---
def get_data_hash():
    """Generate a hash of all PDF files in data/ to detect changes."""
    hasher = hashlib.md5()
    pdf_files = sorted([
        f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")
    ]) if os.path.exists(DATA_DIR) else []

    for fname in pdf_files:
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def load_or_build_vectordb(embedding_model):
    """Load existing ChromaDB or rebuild it if PDFs have changed."""
    os.makedirs(DATA_DIR, exist_ok=True)

    current_hash = get_data_hash()

    # Check if DB exists and data hasn't changed
    if os.path.exists(CHROMA_DIR) and os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            saved = json.load(f)
        if saved.get("hash") == current_hash:
            print("✅ Loading existing ChromaDB (no changes detected)...")
            return Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embedding_model
            )

    # Rebuild DB
    print("🔄 Rebuilding ChromaDB from PDFs...")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{DATA_DIR}'. Please add PDFs and restart.")

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=CHROMA_DIR
    )

    # Save hash so we don't rebuild unnecessarily
    with open(HASH_FILE, "w") as f:
        json.dump({"hash": current_hash}, f)

    print(f"✅ ChromaDB built with {len(chunks)} chunks from {len(pdf_files)} PDF(s).")
    return vectordb


# --- 3. Initialize models ---
print("🔧 Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = load_or_build_vectordb(embedding_model)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.7
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})


# --- 4. Chat function with history + source citations ---
def chat_fn(message, history):
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(message)

        # Build context with source info
        context_parts = []
        sources = set()
        for doc in docs:
            context_parts.append(doc.page_content)
            src = doc.metadata.get("source", "")
            if src:
                sources.add(os.path.basename(src))

        context = "\n\n".join(context_parts)

        # Format conversation history for the prompt
        history_text = ""
        if history:
            for user_msg, bot_msg in history[-4:]:  # last 4 turns
                history_text += f"User: {user_msg}\nAssistant: {bot_msg}\n"

        # Build the full prompt
        prompt = f"""You are Valsu, a warm, supportive, and empathetic mental health assistant. 
Your role is to provide emotional support, psychoeducation, and coping strategies based on the context provided.
Always be compassionate and non-judgmental. If someone is in crisis, always recommend professional help or emergency services.
Never diagnose or prescribe — you are a supportive guide, not a therapist.

Relevant Knowledge:
{context}

Conversation so far:
{history_text}
User: {message}
Valsu:"""

        response = llm.invoke(prompt)
        reply = response.content

        # Append source citations if available
        if sources:
            source_list = ", ".join(sorted(sources))
            reply += f"\n\n📚 *Sources: {source_list}*"

        return reply

    except FileNotFoundError as e:
        return f"⚠️ Setup issue: {str(e)}"
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check your API key and internet connection."


# --- 5. Launch Gradio ---
iface = gr.ChatInterface(
    fn=chat_fn,
    title="🌿 Valsu — Mental Health Assistant",
    description=(
        "Hi, I'm **Valsu** 🌿 — your supportive mental health companion.\n"
        "I'm here to listen, offer coping strategies, and share helpful information.\n"
        "_Remember: I'm not a replacement for professional help. If you're in crisis, please contact a professional._"
    ),
    examples=[
        "I've been feeling really anxious lately, what can I do?",
        "Can you explain what cognitive behavioral therapy is?",
        "I'm having trouble sleeping due to stress. Any tips?",
        "What are some grounding techniques for panic attacks?"
    ],
    theme=gr.themes.Soft(primary_hue="green"),
    retry_btn=None,
    undo_btn="↩ Undo",
    clear_btn="🗑️ Clear Chat"
)

iface.launch(share=True)  # share=True gives a public Gradio URL