from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile, os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Load env
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


# Global paths / vars
FAISS_PATH = "faiss_index"
vectorstore = None
graph = None


# LangGraph State Definition
class QAState(TypedDict):
    question: str
    retrieved_docs: List[str]
    answer: str


# LangGraph Nodes
def retrieve_node(state: QAState) -> QAState:
    """Retrieve relevant chunks from FAISS"""
    global vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["question"])
    return {**state, "retrieved_docs": [d.page_content for d in docs]}


def generate_node(state: QAState) -> QAState:
    """Generate final answer using Gemini"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )
    context = "\n\n".join(state["retrieved_docs"])
    prompt = f"""You are a research assistant. 
        Answer the question based only on the context below.
        Context:
        {context}

        Question: {state['question']}
        Answer:"""
    result = llm.invoke(prompt)
    return {**state, "answer": result.content}


# Build LangGraph
def build_graph():
    workflow = StateGraph(QAState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


# FastAPI Routes
class QueryRequest(BaseModel):
    question: str


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, graph

    # Save PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load + chunk
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Embeddings (Gemini)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
    )

    # Build vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index to disk
    vectorstore.save_local(FAISS_PATH)

    # Build LangGraph
    graph = build_graph()

    return {"message": "PDF uploaded, indexed, and saved locally."}


@app.on_event("startup")
async def load_index_on_startup():
    """Load FAISS index at startup (if exists)"""
    global vectorstore, graph
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
    )
    if os.path.exists(FAISS_PATH):
        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        graph = build_graph()


@app.post("/ask/")
async def ask_question(request: QueryRequest):
    global graph
    if graph is None:
        return {"error": "Please upload a PDF first."}

    # Run graph
    inputs = {"question": request.question, "retrieved_docs": [], "answer": ""}
    final_state = graph.invoke(inputs)
    return {"answer": final_state["answer"]}
