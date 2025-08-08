import requests
import pdfplumber
import docx
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Request, Header
from pydantic import BaseModel
from typing import List, Optional
import json

app = FastAPI()

# --- Models ---
class QueryRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Document Loader ---
def download_file(url):
    """Download file from URL and save locally."""
    response = requests.get(url)
    filename = url.split("?")[0].split("/")[-1]
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

def extract_text_from_pdf(path):
    """Extract text from PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(path):
    """Extract text from DOCX using python-docx."""
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_document(url):
    """Download and extract text from PDF or DOCX."""
    filename = download_file(url)
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(filename)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(filename)
    else:
        raise ValueError("Unsupported file type")

# --- Embedding & Search ---
def chunk_text(text, chunk_size=500):
    """Chunk text into manageable pieces for embedding."""
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n"
        else:
            chunks.append(current)
            current = para + "\n"
    if current:
        chunks.append(current)
    return chunks

def get_embeddings(texts):
    """Get embeddings for a list of texts using a local embedding model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

def build_faiss_index(chunks):
    """Build FAISS index from text chunks."""
    embeddings = get_embeddings(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def search_faiss(index, embeddings, query, chunks, top_k=3):
    """Search FAISS index for top-k relevant chunks."""
    query_emb = get_embeddings([query])[0]
    D, I = index.search(np.array([query_emb]).astype('float32'), top_k)
    return [chunks[i] for i in I[0]]

# --- LLM Clause Matching ---
def llm_decision(query, clauses):
    """Use Phi-2 to answer query based on relevant clauses."""
    prompt = f"Given the following clauses:\n{json.dumps(clauses, indent=2)}\n\nAnswer the query: '{query}'.\nProvide a clear, explainable rationale."
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt from output if repeated
    answer = answer.replace(prompt, "").strip()
    return answer


# --- API Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, Authorization: Optional[str] = Header(None)):
    """
    Main endpoint for intelligent query–retrieval.
    Input:
        - JSON body with fields:
            documents: str (URL to PDF/DOCX)
            questions: List[str] (natural language questions)
        - Example:
            {
                "documents": "https://example.com/policy.pdf",
                "questions": [
                    "Does this policy cover knee surgery, and what are the conditions?",
                    "What is the grace period for premium payment?"
                ]
            }
    Output:
        - JSON response with field:
            answers: List[str] (answers to each question)
        - Example:
            {
                "answers": [
                    "Yes, the policy covers knee surgery under certain conditions...",
                    "A grace period of thirty days is provided for premium payment..."
                ]
            }
    1. Loads document
    2. Chunks and embeds text
    3. Retrieves relevant clauses
    4. Uses LLM for decision and rationale
    5. Returns structured JSON answers
    """
    # 1. Load document
    text = load_document(request.documents)
    # 2. Chunk and embed
    chunks = chunk_text(text)
    index, embeddings = build_faiss_index(chunks)
    answers = []
    for q in request.questions:
        # 3. Retrieve relevant clauses
        relevant_clauses = search_faiss(index, embeddings, q, chunks)
        # 4. LLM decision
        answer = llm_decision(q, relevant_clauses)
        answers.append(answer)
    return QueryResponse(answers=answers)

# --- Documentation ---
"""
API Documentation:
POST /hackrx/run
Authorization: Bearer <token>
Request Body:
{
    "documents": "<PDF/DOCX URL>",
    "questions": ["<query1>", "<query2>", ...]
}
Response:
{
    "answers": ["<answer1>", "<answer2>", ...]
}
Each answer is generated by retrieving relevant clauses and using GPT-4 for explainable rationale.
"""


