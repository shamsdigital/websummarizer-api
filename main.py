# websummarizer-api/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

app = FastAPI()

# 🧠 Replace with your OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-REPLACE_WITH_YOUR_KEY")

class URLRequest(BaseModel):
    url: str

class SearchQuery(BaseModel):
    query: str

# 🔍 Crawl the website
def crawl_website(base_url):
    visited = set()
    to_visit = [base_url]
    pages = {}

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            pages[url] = text

            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link['href'])
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    to_visit.append(full_url)
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            continue
    return pages

# 📚 Text chunking
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# 📦 Save to FAISS
def save_embeddings(text_chunks, path="faiss_store"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local(path)

# 🔍 Load FAISS and search
def search_memory(query, path="faiss_store"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(path, embeddings)
    results = db.similarity_search(query)
    return [r.page_content for r in results]

# 🧠 OpenRouter Summarizer
def query_llm(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that summarizes websites."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
    return response.json()["choices"][0]["message"]["content"]

# ✅ POST: Crawl, summarize and store
@app.post("/summarize")
def summarize(request: URLRequest):
    url = request.url
    pages = crawl_website(url)
    all_text = "\n".join(pages.values())
    chunks = chunk_text(all_text)
    save_embeddings(chunks)

    summary = query_llm(f"Summarize the website at {url}. Include what it's about and any contact info:\n\n{all_text[:4000]}")
    return {"summary": summary, "total_chunks": len(chunks)}

# 🔎 GET: Search from memory
@app.get("/search")
def search(query: str = Query(...)):
    results = search_memory(query)
    return {"matches": results[:5]}  # Limit to top 5
