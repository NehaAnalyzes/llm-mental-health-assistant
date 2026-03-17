# 🌿 Mental Health Assistant - RAG-based AI Support System

A production-style Retrieval-Augmented Generation (RAG) application that provides supportive mental health responses using domain documents and a large language model.

This project demonstrates real world GenAI system design including vector search, document retrieval, prompt orchestration, and cloud deployment.

---

## 📌 Overview

This system combines:

- Semantic document retrieval (RAG)
- LLM based response generation
- Mental health knowledge grounding
- Real time conversational UI

The assistant provides supportive, context-aware responses by retrieving relevant content from curated mental health documents.

---

## 🎯 Problem Statement

Generic LLM chatbots often:

- Hallucinate mental health advice
- Provide unsafe or ungrounded responses
- Lack domain grounding
- Cannot scale safely for sensitive topics

This project addresses these issues by:

✅ Using Retrieval Augmented Generation  
✅ Grounding responses in verified documents  
✅ Designing ethical AI interaction patterns  
✅ Deploying a real time AI support prototype  


---

## 🚀 Live Demo

🔗 https://llm-mental-health-assistant-k9pzq9d9oyqzjwhkbytzep.streamlit.app/
---


---

## 🧠 System Architecture

User Query  
→ Embedding Generation  
→ Vector Search (Chroma DB)  
→ Relevant Context Retrieval  
→ Prompt Construction  
→ LLM Response (Groq LLaMA Model)  
→ Streamlit Chat Interface  

---

## ⚙️ Tech Stack

### 🔹 GenAI & NLP
- LangChain
- Groq LLM (LLaMA-3.3)
- Sentence Transformers

### 🔹 Vector Database
- ChromaDB

### 🔹 Backend Logic
- Python

### 🔹 UI & Deployment
- Streamlit Cloud

### 🔹 Document Processing
- PyPDF

---

## 📊 Key Features

✔ Retrieval-Augmented Generation pipeline  
✔ Precomputed vector database for fast startup  
✔ Context-aware mental health responses  
✔ Free LLM integration (Groq)  
✔ Cloud-deployed real-time chatbot  
✔ Scalable document ingestion architecture  
✔ Ethical AI response framing  


---

## 🧩 RAG Pipeline Details

### 1️⃣ Document Processing
- Mental health PDFs ingested
- Recursive chunking strategy
- Semantic embedding generation

### 2️⃣ Retrieval System
- Vector similarity search using Chroma
- Top k context selection
- Context compression in prompt

### 3️⃣ LLM Response Generation
- LLaMA-3 via Groq inference
- Controlled temperature for safe responses
- Context grounded generation

---

## ⚡ Performance Optimizations

- Prebuilt vector database (reduces cold start time)
- Lightweight embedding model
- Reduced retrieval depth
- Efficient Streamlit caching
- Cloud friendly architecture

---

## 🛡️ Ethical Considerations

This system is designed as:

⚠️ A supportive informational tool  
❌ NOT a replacement for professional therapy  

Future safety improvements planned:

- Crisis detection layer
- Response guardrails
- Citation highlighting
- Safety classifier integration

---

## 📈 Future Improvements

- Multi LLM fallback system
- Response citation display
- Retrieval evaluation metrics
- Memory augmented conversations
- Fine-tuned domain embedding model
- Clinical safety framework

---

## 🎓 Learning Outcomes

Through this project, I gained experience in:

- Real world GenAI system deployment
- Vector database optimization
- LLM orchestration strategies
- Cloud debugging & dependency resolution
- Ethical AI system design

---



## ⭐ Acknowledgements

- LangChain ecosystem
- Groq inference platform
- Streamlit Cloud
- Open mental health resources
