# 📘 Semantic Search Bot

A full-stack semantic search application that allows users to upload documents and query them using natural language. Powered by **transformer embeddings**, **vector databases**, and **LangChain**, this tool retrieves highly relevant information by understanding context, not just keywords.

---

## 🚀 Features

- 🔍 **Semantic Search:** Retrieves answers based on meaning, not just keyword matches.
- 🧠 **Transformer Embeddings:** Uses pre-trained language models to encode documents and queries.
- 📁 **Multi-Document Support:** Upload `.txt`, `.pdf`, or `.docx` files to build your own knowledge base.
- 🗂 **FAISS Vector Store:** Efficient similarity search over large document sets.
- 🧵 **LangChain Pipelines:** Handles chunking, embedding, and query workflows.
- 💻 **React Frontend:** Clean, interactive UI for uploading files and querying in real-time.

---

## 🛠 Tech Stack

| Layer         | Tools/Frameworks                              |
|---------------|-----------------------------------------------|
| **Frontend**  | React, Tailwind CSS, Axios                    |
| **Backend**   | Python, LangChain, FAISS, Hugging Face        |
| **Embeddings**| `sentence-transformers/all-MiniLM-L6-v2` (or configurable) |
| **File Support** | TXT, PDF, DOCX                            |

---

## 📂 Folder Structure (Simplified)

```
semantic-search-bot/
├── backend/
│   ├── app.py
│   ├── utils/
│   └── vector_store/
├── data/
├── semantic-search-frontend/
│   ├── public/
│   └── src/
└── README.md
```

---

## ⚙️ How It Works

1. **Upload Files** → User uploads documents through the frontend.
2. **Chunking & Embedding** → LangChain splits the text and creates vector embeddings.
3. **Vector Store** → FAISS stores the vectors for fast retrieval.
4. **Semantic Query** → User submits a query, which is embedded and matched with relevant chunks.
5. **Answer Returned** → Most relevant text snippets are shown to the user.

---

## 📦 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/semantic-search-bot.git
cd semantic-search-bot
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

### 3. Frontend Setup

```bash
cd semantic-search-frontend
npm install
npm start
```

### 4. Visit:

```
http://localhost:3000
```

---

## 📌 TODOs / Improvements

- [ ] Add multi-language support
- [ ] Enable answer summarization
- [ ] Add model switching from UI
- [ ] Deploy to Hugging Face Spaces or Vercel

---
