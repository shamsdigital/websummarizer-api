# websummarizer-api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

app = FastAPI()

class URLRequest(BaseModel):
    url: str

# Function to crawl the website and extract text
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

# POST endpoint to summarize the website
@app.post("/summarize")
def summarize(request: URLRequest):
    url = request.url
    pages = crawl_website(url)
    if not pages:
        raise HTTPException(status_code=404, detail="No content found at the provided URL.")
    
    all_text = "\n".join(pages.values())
    summary = all_text[:4000]  # Simple summary: first 4000 characters
    return {"summary": summary, "total_chunks": len(pages)}

# GET endpoint to check if the service is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Web Summarizer API!"}
