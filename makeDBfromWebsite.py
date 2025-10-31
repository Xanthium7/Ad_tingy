

from __future__ import annotations
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import argparse

import os
import re
import time
import hashlib
import urllib.parse
from collections import deque
from typing import Set, List

import requests
from bs4 import BeautifulSoup

# Disable TensorFlow path in transformers to avoid Keras 3 incompat issue (we only need PyTorch)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

START_URL = "https://vegeta.com/en"
ALLOWED_PREFIX = "https://vegeta.com/en"  # Restrict to English section
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
MAX_PAGES = 300  # Safety limit; increase if needed
MAX_DEPTH = 6    # Prevent infinite crawl loops
REQUEST_DELAY = 0.5  # Seconds between requests (politeness)
TIMEOUT = 15

PERSIST_DIR = "./vegeta/chroma_db_nccn"
EXTRA_MARKDOWN_PATHS = ["./vegeta_doc/queries.md"]


def normalize_url(url: str) -> str:
    # Remove fragments, normalize trailing slash
    parsed = urllib.parse.urlsplit(url)
    cleaned = parsed._replace(fragment="", query=parsed.query).geturl()
    if cleaned.endswith('/') and len(cleaned) > len("https://vegeta.com/") and not parsed.path.endswith('/'):
        # Keep as-is if original path didn't have slash
        pass
    return cleaned.rstrip('/')


def is_valid_link(href: str) -> bool:
    if not href:
        return False
    if href.startswith('mailto:') or href.startswith('tel:'):
        return False
    if any(href.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.pdf', '.zip', '.mp4', '.mp3')):
        return False
    return True


def extract_text(html: str, url: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    # Remove unwanted tags
    for tag in soup(['script', 'style', 'noscript', 'svg', 'iframe', 'header', 'footer', 'form', 'nav']):
        tag.decompose()
    # Sometimes content hides in repetitive whitespace; collapse it
    text = soup.get_text(separator=' ')  # keep some separation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def crawl_site(start_url: str, allowed_prefix: str) -> List[Document]:
    queue = deque([(start_url, 0)])
    seen: Set[str] = set()
    docs: List[Document] = []
    content_hashes: Set[str] = set()

    session = requests.Session()
    session.headers.update(
        {'User-Agent': USER_AGENT, 'Accept-Language': 'en-US,en;q=0.9'})

    while queue and len(seen) < MAX_PAGES:
        url, depth = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        try:
            resp = session.get(url, timeout=TIMEOUT)
            if not resp.ok or 'text/html' not in resp.headers.get('Content-Type', ''):
                continue
            html = resp.text
            text = extract_text(html, url)
            if not text or len(text.split()) < 20:
                continue  # skip near-empty pages
            # Deduplicate by content hash
            h = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if h in content_hashes:
                continue
            content_hashes.add(h)
            docs.append(Document(page_content=text, metadata={
                        'source': url, 'url': url}))
            print(f"[PAGE {len(docs)}] {url} (words={len(text.split())})")

            if depth < MAX_DEPTH:
                soup = BeautifulSoup(html, 'html.parser')
                for a in soup.find_all('a', href=True):
                    href = a['href'].strip()
                    if not is_valid_link(href):
                        continue
                    next_url = urllib.parse.urljoin(url, href)
                    if next_url.startswith(allowed_prefix):
                        next_url = normalize_url(next_url)
                        if next_url not in seen:
                            queue.append((next_url, depth + 1))
        except requests.RequestException as e:
            print(f"[ERROR] {url}: {e}")
        time.sleep(REQUEST_DELAY)

    return docs


def load_markdown_documents(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for path in paths:
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            print(f"[WARN] Markdown source not found: {path}")
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except OSError as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
            continue
        if not content:
            print(f"[WARN] Markdown source empty: {path}")
            continue
        # Preserve relative path in metadata for transparency
        rel_path = os.path.relpath(abs_path)
        docs.append(Document(page_content=content, metadata={
            'source': rel_path.replace('\\', '/'),
            'url': rel_path.replace('\\', '/'),
            'source_type': 'markdown'
        }))
        print(f"[MD] Loaded {rel_path}")
    return docs


def build_vectorstore(docs: List[Document], *, rebuild: bool = True): # '*'  basically means to force keyword args after this
    if not docs:
        raise ValueError("No documents collected from crawl.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    print(f"Total original pages: {len(docs)} | Chunks: {len(split_docs)}")

    # Prepare embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}
    )

    if rebuild:
        # Clear existing directory contents for a clean rebuild
        if os.path.isdir(PERSIST_DIR):
            for root, dirs, files in os.walk(PERSIST_DIR):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except OSError:
                        pass
        os.makedirs(PERSIST_DIR, exist_ok=True)
        vectorstore = Chroma.from_documents(
            split_docs, embedding_function, persist_directory=PERSIST_DIR)
        print(
            f"Vector store rebuilt. Total embeddings: {vectorstore._collection.count()}")
        return vectorstore

    # Incremental update path: reuse existing store and append new chunks
    if not os.path.isdir(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        print(
            "[WARN] Existing vector store not found or empty; performing full rebuild instead.")
        return build_vectorstore(docs, rebuild=True)

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR, embedding_function=embedding_function)

    sources_to_refresh = {
        doc.metadata.get('source')
        for doc in split_docs
        if doc.metadata.get('source')
    }
    for source in sources_to_refresh:
        vectorstore.delete(where={'source': source})

    vectorstore.add_documents(split_docs)
    vectorstore.persist()
    print(
        f"Vector store updated incrementally. Total embeddings: {vectorstore._collection.count()}")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(
        description="Build or update the Vegeta knowledge base.")
    parser.add_argument(
        "--markdown-only",
        action="store_true",
        help="Skip website crawl and append only markdown documents to the existing vector store."
    )
    args = parser.parse_args()

    docs: List[Document] = []

    if not args.markdown_only:
        print(
            f"Starting crawl at {START_URL}\nMax pages: {MAX_PAGES} | Max depth: {MAX_DEPTH}")
        docs.extend(crawl_site(START_URL, ALLOWED_PREFIX))
    else:
        print("Skipping website crawl (markdown-only mode).")

    markdown_docs = load_markdown_documents(EXTRA_MARKDOWN_PATHS)
    if markdown_docs:
        docs.extend(markdown_docs)
        print(f"Markdown documents added: {len(markdown_docs)}")

    if not docs:
        print("No documents to process. Exiting.")
        return

    print("Building vector store...")
    build_vectorstore(docs, rebuild=not args.markdown_only)
    print("Done.")


if __name__ == "__main__":
    main()
