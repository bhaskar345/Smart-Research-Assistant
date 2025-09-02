# 📚 Smart Research Assistant

A full-stack AI-powered research assistant that lets you **upload PDFs**, **ask questions**, and get **Gemini-powered answers** using **LangGraph** for workflow orchestration.

---

## 🚀 Features
- Upload and analyze PDFs
- Chat with AI about the uploaded content
- Uses **LangGraph** + **Gemini API**
- Backend: **FastAPI**
- Frontend: **Bootstrap + Vanilla JS**
- Local saving/loading of documents

---

## ⚙️ Project Setup

### Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate    # On Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Add .env
```bash
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

### Run FastAPI Server
```bash
uvicorn app:app --reload --port 8000
```
Server runs at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🖥️ Usage
1. Start the backend (`uvicorn app:app --reload`).
2. Open `http://127.0.0.1:8000` in your browser.
3. Upload a PDF file.
4. Ask questions in the chat.
5. Get AI-powered answers!

---